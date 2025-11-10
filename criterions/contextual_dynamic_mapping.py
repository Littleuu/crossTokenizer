import logging
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
from typing import Dict, List
from .various_divergence import VariousDivergence
import copy
import json
import inspect

TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
}

def calculate_weight(logits):
    probabilities = torch.softmax(logits, dim=-1)
    epsilon = 1e-10
    probabilities = probabilities + epsilon
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=-1)
    entropy_min = entropy.min()
    entropy_max = entropy.max()
    factor = torch.sigmoid((entropy - entropy_min)/(entropy_max-entropy_min) * 4 - 2) # [0,1]\
    factor = torch.tensor((factor * 3 + 3), dtype=torch.int32).detach().cpu().tolist()

    return factor

class CDMKLD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super(CDMKLD, self).__init__(args, padding_id=padding_id)
        self.args = args
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.simi_threadshold = 0.3
        self.topk = 100
    
    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.teacher_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]
        self.distiller = distiller
        
        if self.args.teacher_to_student_id_mapping is not None:
            self.tea2stu_id_mapping = json.load(open(self.args.teacher_to_student_id_mapping))
                    
            self.stu2tea_id_mapping = torch.zeros(distiller.student_tokenizer.vocab_size+256, dtype=torch.long)
            for tea_id in self.tea2stu_id_mapping:

                if self.tea2stu_id_mapping[tea_id]!=0:
                    self.stu2tea_id_mapping[self.tea2stu_id_mapping[tea_id]] = int(tea_id)

            self.tea2stu_id_mapping = list(self.tea2stu_id_mapping.values())
            tea_vocab_size = self.teacher_tokenizer.vocab_size + len(self.teacher_tokenizer.added_tokens_decoder)
            if len(self.tea2stu_id_mapping) != tea_vocab_size:
                self.tea2stu_id_mapping += [0] * (tea_vocab_size - len(self.tea2stu_id_mapping))
            self.tea2stu_id_mapping = torch.LongTensor(self.tea2stu_id_mapping).to(model.device)
            self.stu2tea_id_mapping = torch.LongTensor(self.stu2tea_id_mapping).to(model.device)
            self.stu2tea_id_mapping_tea = torch.LongTensor(torch.arange(self.stu2tea_id_mapping.shape[0])).to(model.device)
            self.stu2tea_id_mapping_stu = copy.deepcopy(self.stu2tea_id_mapping)
            self.em_tea2stu_id_mapping = copy.deepcopy(self.tea2stu_id_mapping)
            self.em_stu2tea_id_mapping = copy.deepcopy(self.stu2tea_id_mapping)

        if "position_ids" in inspect.signature(model.forward).parameters:
            outputs = model(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                position_ids=input_data.get("position_ids", None),
                output_hidden_states=True
            )
        else:
            outputs = model(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                output_hidden_states=True
            )
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]
        log["nll_loss"] = ce_loss

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        edit_kd_loss, unmasked_rate = self.compute_edit_distance_kd_loss(outputs, teacher_outputs, input_data, output_data, distiller)
        log["kd_loss"] = edit_kd_loss

        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * edit_kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, 
            output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            log
        )

        return loss / batch_denom, logging_output

    def compute_edit_distance_kd_loss(
            self, outputs_student, outputs_teacher, inputs, output_data, distiller
        ):
            target = output_data["label"]
            teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
            student_logits = outputs_student.logits
            stu_tokenizer = distiller.student_tokenizer
            tea_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]

            bsz = target.shape[0]
            aligned_tea_logits = []
            aligned_stu_logits = []
            for i in range(bsz):
                assert self.padding_id in target[i]
                stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
                stu_input_ids = inputs["input_ids"][i, stu_content_idx]

                tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
                tea_input_ids = inputs[f"teacher_{distiller.teacher_model_type}_input_ids"][i, tea_content_idx]
                
                stu_per_step_logits = student_logits[i, stu_content_idx, :]
                tea_per_step_logits = outputs_teacher.logits[i, tea_content_idx, :]
                if stu_per_step_logits.shape[-1]==0 or tea_per_step_logits.shape[-1]==0:
                    return torch.Tensor([0.0]).to(student_logits.device), 0.0
                
                aligned_tea_content_per_step_logits, meaned_stu_content_logits, unmask_rate = self.transform_step_logits_fast(
                    stu_tokenizer,
                    tea_tokenizer,
                    stu_input_ids,
                    stu_per_step_logits,
                    tea_input_ids,
                    tea_per_step_logits,
                )
                
                aligned_stu_logits.append(meaned_stu_content_logits)
                aligned_tea_logits.append(aligned_tea_content_per_step_logits)
                assert meaned_stu_content_logits.shape == aligned_tea_content_per_step_logits.shape
            
            aligned_tea_logits = torch.cat(aligned_tea_logits, 0)
            aligned_stu_logits = torch.cat(aligned_stu_logits, 0)
            in_len = aligned_stu_logits.shape[1]
            kd_loss = self.kl_dist_func(
                aligned_stu_logits, 
                aligned_tea_logits, 
                output_data["label"],
                reduction='mean'
            )
            
            return kd_loss, unmask_rate
    
    def kl_dist_func(
            self, 
            logits, 
            teacher_logits, 
            target=None,
            reduction=None
        ):
            lprobs = torch.log_softmax(logits/self.kd_temp, -1, dtype=torch.float32)
            teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
            teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32) * (self.kd_temp ** 2)
            kld = (teacher_probs * (teacher_lprobs - lprobs))
            inf_mask = logits.isinf()
            kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
            if reduction == "sum":
                kld = kld.sum()
            else:
                kld = kld.mean()

            return kld
    
    def merge_tensor(self, values, mapping_list):
            merged_values = []

            for ids in mapping_list:
                merged_values.append(values[ids].mean(dim=0))
            merged_values = torch.stack(merged_values, dim=0)
            return merged_values

    def transform_step_logits_fast(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_input_ids: torch.LongTensor,
        base_model_per_step_logits: torch.FloatTensor,
        blending_model_input_ids: torch.LongTensor,
        blending_model_per_step_logits: torch.FloatTensor,
    ):
        """faster implementation to align logits"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        base_model_tokens = [base_model_tokenizer.convert_tokens_to_string([tok]) for tok in base_model_tokens]
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        blending_model_tokens = [blending_model_tokenizer.convert_tokens_to_string([tok]) for tok in blending_model_tokens]
        if base_model_tokenizer.__class__ not in TOKENIZER_TO_SPECIAL_TOKEN:
            base_model_special_token = "Ġ"
        else:
            base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
                base_model_tokenizer.__class__
            ]
        if blending_model_tokenizer.__class__ not in TOKENIZER_TO_SPECIAL_TOKEN:
            blending_model_special_token = '_'
        else:
            blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
                blending_model_tokenizer.__class__
            ]
            
        specTok_mapper = {
                '</s>': '<|im_end|>',
                '<|endoftext|>':'<|endoftext|>'
            }

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            if a in specTok_mapper and b in specTok_mapper.values():
                return 0.0
            if b in specTok_mapper and a in specTok_mapper.values():
                return 0.0
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            aa = a.replace(" ", "")
            bb = b.replace(" ", "")
            dist = editdistance.eval(aa, bb) 
            if len(aa)==len(bb)==0:
                return 0.0
            dist = dist / (len(aa)+len(bb))
            return dist
        
        def cost_fn(a, b):
            """cost function for sequence alignment"""
            if a in specTok_mapper and b in specTok_mapper.values():
                return 0.0
            if b in specTok_mapper and a in specTok_mapper.values():
                return 0.0
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            aa = a.replace(" ", "")
            bb = b.replace(" ", "")
            dist = editdistance.eval(aa, bb)
            return dist

        blending_dist_factor = calculate_weight(blending_model_per_step_logits) 
        base_dist_factor = calculate_weight(base_model_per_step_logits)
        # obtain sequence token alignment (each stu token to which tea token)
        _, _, blending_to_base, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, blending_dist_factor, base_dist_factor, norm_func=cost_fn
        ) 

        merged_blending_tokens = []
        for ids in base_to_blending:
            merged_token = ''
            for id in ids:
                merged_token += blending_model_tokens[id]
            merged_blending_tokens.append(merged_token)

        blending_model_per_step_logits = self.merge_tensor(
            blending_model_per_step_logits,
            base_to_blending
        )
        cnt_merge_blending_to_base = []
        for ids in blending_to_base:
            if ids not in cnt_merge_blending_to_base:
                cnt_merge_blending_to_base.append(ids)
        
        blending_model_per_step_logits = self.merge_tensor(
            blending_model_per_step_logits,
            cnt_merge_blending_to_base
        )
        base_model_per_step_logits = self.merge_tensor(
            base_model_per_step_logits,
            cnt_merge_blending_to_base
        )
        
        topK = self.topk
        blending_topk_ids = torch.topk(blending_model_per_step_logits, topK).indices
        base_topk_ids = torch.topk(base_model_per_step_logits, topK).indices
        
        blending_topk_tokens = []
        for ids in blending_topk_ids:
            blending_topk_tokens.append([blending_model_tokenizer.decode(id) for id in ids])
        
        base_topk_tokens = []
        for ids in base_topk_ids:
            base_topk_tokens.append([base_model_tokenizer.decode(id) for id in ids])

        tea2stu_mapper = self.tea2stu_id_mapping
        def get_dymaic_mapper(
                blending_topk_ids, 
                base_topk_ids, 
                blending_topk_tokens, 
                base_topk_tokens, 
                blending2base_mapper=tea2stu_mapper,
                em_mapper=self.em_tea2stu_id_mapping,
            ):
            dist_threashold = self.simi_threadshold
            # get the exact matching result
            em_converted_base_topk_ids = blending2base_mapper[blending_topk_ids]
            # get the elements that are not exact match use a mask with 0 judgement
            miss_hit_mask = torch.eq(em_converted_base_topk_ids, 0)
            # get the unmapped base tokens, and the correspondent candidate tokens in teacher
            unmapped_blending_list = []
            # [base_topk_ids[pos] for pos in torch.nonzero(miss_hit_mask)]
            for pos in torch.nonzero(miss_hit_mask): unmapped_blending_list.append(blending_topk_ids[pos[0]][pos[1]])

            unmapped_blending_tokens = [blending_topk_tokens[pos[0]][pos[1]] for pos in torch.nonzero(miss_hit_mask)]
            candidate_list = [base_topk_ids[pos[0]] for pos in torch.nonzero(miss_hit_mask)]
            candidate_tokens = [base_topk_tokens[pos[0]] for pos in torch.nonzero(miss_hit_mask)]
            # traversal to get the supplemental mapping pairs.
            matched_ids = torch.nonzero(torch.eq(blending2base_mapper.squeeze(0), 0)).reshape(-1).tolist()
            matched_set = set(matched_ids)
            new_mapper = {}
            unmatch_list = []
            if dist_threashold >0.0001:
                for id, token, cand_ids, cand_toks in zip(unmapped_blending_list, unmapped_blending_tokens, candidate_list, candidate_tokens):
                    # if the token is already mapped in Exact Matching Mode, skip
                    if em_mapper[id]!=0:
                        continue
                    cand_ids = cand_ids.tolist()
                    cand_mapper = {tid:tok for tok, tid in zip(cand_toks, cand_ids)}
                    cand_ids = list(set(cand_ids).difference(matched_set))
                    if len(cand_ids)==0:
                        continue
                    min_dist = 1000
                    simi_id = 0
                    for cand_id in cand_ids:
                        cand_tok = cand_mapper[cand_id]
                        tok_dist = dist_fn(token, cand_tok)
                        if tok_dist<dist_threashold and tok_dist<min_dist:
                            simi_id = cand_id
                            min_dist = tok_dist
                    if simi_id!=0:
                        # update the mapper, keep the life cycle in the whole training step
                        blending2base_mapper[id] = simi_id
                        new_mapper[token] = cand_mapper[simi_id]
                    else:
                        unmatch_list.append(token)

                    # import pdb;pdb.set_trace()
                # print(new_mapper)
                # print(unmatch_list)
                # exit(0)
            converted_base_topk_ids = blending2base_mapper[blending_topk_ids].to(blending_model_per_step_logits.device)
            unmatch_mask = torch.eq(converted_base_topk_ids, 0)
            masked_blending_topk_ids = blending_topk_ids.masked_fill_(unmatch_mask, 0)
            return converted_base_topk_ids, masked_blending_topk_ids

        # this block, convert the student token id to map the teacher top 100
        base_logits = []
        blending_logits = []
        stu_converted_topk_ids, tea_converted_topk_ids = get_dymaic_mapper(
            blending_topk_ids, 
            base_topk_ids, 
            blending_topk_tokens, 
            base_topk_tokens, 
            blending2base_mapper=copy.deepcopy(self.tea2stu_id_mapping),
            em_mapper=self.em_tea2stu_id_mapping,
            )
        
        stu_model_per_step_logits = base_model_per_step_logits.gather(-1, stu_converted_topk_ids)
        tea_model_per_step_logits = blending_model_per_step_logits.gather(-1, tea_converted_topk_ids)

        stu_logit_mask = stu_converted_topk_ids.eq(0)            
        stu_model_per_step_logits.masked_fill_(stu_logit_mask, -10000.0)
        tea_logit_mask = tea_converted_topk_ids.eq(0)            
        tea_model_per_step_logits.masked_fill_(tea_logit_mask, -10000.0)
        mask_rate = stu_logit_mask.sum().item() / (stu_logit_mask.size(0) * stu_logit_mask.size(1))
        
        base_logits.append(tea_model_per_step_logits)
        blending_logits.append(stu_model_per_step_logits)

        # another direction
        tea_converted_topk_ids, stu_converted_topk_ids = get_dymaic_mapper(
            base_topk_ids, 
            blending_topk_ids, 
            base_topk_tokens, 
            blending_topk_tokens, 
            blending2base_mapper=copy.deepcopy(self.stu2tea_id_mapping),
            em_mapper=self.em_stu2tea_id_mapping
            )
        stu_model_per_step_logits = base_model_per_step_logits.gather(-1, stu_converted_topk_ids)
        tea_model_per_step_logits = blending_model_per_step_logits.gather(-1, tea_converted_topk_ids)
        stu_logit_mask = stu_converted_topk_ids.eq(0)            
        stu_model_per_step_logits.masked_fill_(stu_logit_mask, -10000.0)
        tea_logit_mask = tea_converted_topk_ids.eq(0)            
        tea_model_per_step_logits.masked_fill_(tea_logit_mask, -10000.0)
        mask_rate += stu_logit_mask.sum().item() / (stu_logit_mask.size(0) * stu_logit_mask.size(1))
        mask_rate = mask_rate/2
        base_logits.append(stu_model_per_step_logits)
        blending_logits.append(tea_model_per_step_logits)
        
        return torch.cat(base_logits, dim=-1), torch.cat(blending_logits, dim=-1), 1-mask_rate


    def transform_step_logits(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_vocab: Dict[str, int],
        base_model_input_ids: List[int],
        blending_model_input_ids: List[int],
        blending_model_per_step_logits: List[List[float]],
        blending_model_per_step_indices: List[List[int]],
        vocab_align_type: str = "hard",
        blending_to_base_mapping: Dict[str, str] = None,
    ):
        """Align blending model per step logits & indices with base model. (original implementation in FuseLLM)"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        )
        aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
            [],
            [],
        )
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(
                    blending_model_special_token, base_model_special_token
                )
                if (
                    (
                        blending_model_tokenizer.__class__
                        == transformers.GPTNeoXTokenizerFast
                        or blending_model_tokenizer.__class__
                        == transformers.GPT2TokenizerFast
                    )
                    and i == 0
                    and base_token.startswith(base_model_special_token)
                    and not blending_token.startswith(base_model_special_token)
                ):
                    blending_token = (
                        base_model_special_token + blending_token
                    )  # special case for mpt
                if vocab_align_type == "hard":
                    if (
                        base_token == blending_token
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                elif vocab_align_type == "soft":
                    if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping
                        and base_token == blending_to_base_mapping[blending_token]
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):  
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            blending_t = blending_to_base_mapping[blending_t]
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                            else:
                                logging.warning(
                                    f"blending_t: {blending_t} not in base_model_vocab!"
                                )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                else:
                    logging.warning(
                        f"The vocab_align_type: '{vocab_align_type}' is not support!"
                    )
                    raise NotImplementedError
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(
                aligned_blending_model_per_step_index
            )
            aligned_blending_model_per_step_logits.append(
                aligned_blending_model_per_step_logit
            )
        return (
            aligned_blending_model_per_step_logits,
            aligned_blending_model_per_step_indices,
        )
    
    def dtw(self, series_1, series_2, series1_factor, series2_factor, norm_func=np.linalg.norm):
        matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
        matrix[0, :] = np.inf
        matrix[:, 0] = np.inf
        matrix[0, 0] = 0
        for i, (vec1, fc1) in enumerate(zip(series_1, series1_factor)):
            for j, (vec2, fc2) in enumerate(zip(series_2, series2_factor)):
                cost = norm_func(vec1, vec2) * fc1 * fc2
                
                # cost = norm_func(vec1, vec2)
                matrix[i + 1, j + 1] = cost + min(
                    matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
                )
        matrix = matrix[1:, 1:]
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [list() for v in range(matrix.shape[0])]
        mappings_series_2 = [list() for v in range(matrix.shape[1])]
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()

        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
