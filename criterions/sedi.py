import torch
from .cross_entropy_loss import CrossEntropyLoss
import torch.nn.functional as F
import networkx as nx
from scipy.optimize import linear_sum_assignment
from geomloss import SamplesLoss
import ot
from torch import nn
from collections import defaultdict
from scipy.sparse import coo_matrix
from .various_divergence import VariousDivergence
from transformers import PreTrainedTokenizer
from typing import List, Tuple
import transformers
import re
import inspect

specialtokens = {
    "gpt2": "Ġ", 
    "llama": "▁",
    "qwen": "Ġ",
    "deepseek": "Ġ",
    "pythia": "Ġ",
    "opt": "Ġ",
    }

def merge_alignments_simple(alignment):
    merged = []

    for s_idx, t_indices in alignment:
        s_set = set([s_idx])
        t_set = set(t_indices)

        merged_into = []
        remaining = []

        for s_group, t_group in merged:
            if s_set & s_group or t_set & t_group:
                s_set |= s_group
                t_set |= t_group
                merged_into.append((s_group, t_group))
            else:
                remaining.append((s_group, t_group))

        remaining.append((s_set, t_set))
        merged = remaining

    return [(sorted(list(s)), sorted(list(t))) for s, t in merged]

def get_token_spans(text: str, id_list: List[int], tokenizer: PreTrainedTokenizer) -> List[Tuple[int, int]]:
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = encoding["offset_mapping"]
    ids = encoding["input_ids"]
    
    if len(offsets) > len(id_list):
        offsets = offsets[:len(id_list)]
        ids = encoding["input_ids"][:len(id_list)]
    return offsets, ids

def align_tokenizers(text: str,
                     student_tokenizer: PreTrainedTokenizer,
                     teacher_tokenizer: PreTrainedTokenizer,
                     student_ids: List[int],
                     teacher_ids: List[int]) -> List[Tuple[List[int], List[int]]]:

    student_spans, stu_ids_ = get_token_spans(text, student_ids, student_tokenizer)
    student_id_len = len(stu_ids_)            
    teacher_spans, tea_ids_ = get_token_spans(text, teacher_ids, teacher_tokenizer)
    teacher_id_len = len(tea_ids_)            
    
    assert student_spans[-1][1] == teacher_spans[-1][1]
        
    alignment = []

    def span_overlaps(s_start, s_end, t_start, t_end):
        if t_start == t_end:
            return s_start <= t_start < s_end
        if s_start == s_end:
            return t_start <= s_start < t_end
        return not (t_end <= s_start or t_start >= s_end)

    for i, (s_start, s_end) in enumerate(student_spans):
        matching_teacher = []

        k = 0
        while k < len(teacher_spans):
            t_start, t_end = teacher_spans[k]
            if span_overlaps(s_start, s_end, t_start, t_end):
                matching_teacher.append(k)
            k += 1

        if matching_teacher:
            alignment.append((i, matching_teacher))

    assert alignment[-1][0] == (len(stu_ids_)-1) and alignment[-1][1][-1] == (len(tea_ids_)-1)

    return alignment, student_id_len, teacher_id_len


def detect_next_token(K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, tea_idx, next_stu_ids, next_tea_ids, student_per_step_logit, teacher_logits, teacher_logits_projected, align_idx0, teacher_special_token, student_special_token, start=False):
    back_logit = student_per_step_logit if student_per_step_logit.size(0) > teacher_logits_projected.size(0) else teacher_logits_projected
    if len(next_stu_ids) == 1 and len(next_tea_ids) == 1:
        if start:
            aligned_vec = torch.full_like(back_logit[next_tea_ids[0]], mask_id)
            topk_values, topk_indices = torch.topk(align_idx0[0], k=K, dim=-1)
            token_ids = topk_indices.tolist()
            scores = topk_values.tolist()
            for j in range(K):
                tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
                if tea_token != teacher_special_token:
                    tea_token = re.sub(f"^{teacher_special_token}", "", tea_token)
                else:
                    tea_token = re.sub(f"^{teacher_special_token}", student_special_token, tea_token)
                corres_stu_token_id = tokenizer_A.convert_tokens_to_ids(tea_token)
                if corres_stu_token_id != tokenizer_A.unk_token_id:
                    if aligned_vec[corres_stu_token_id] == mask_id:
                        aligned_vec[corres_stu_token_id] = scores[j]
                else:
                    new_id = tokenizer_A.encode(tea_token, add_special_tokens=False)
                    if aligned_vec[new_id[0]] == mask_id:
                        aligned_vec[new_id[0]] = scores[j]
            tokenizer_aligned_teacher_logit = align_idx0[0]
        else:
            aligned_vec = teacher_logits_projected[tea_idx]  
            tokenizer_aligned_teacher_logit = teacher_logits[tea_idx] 
    elif len(next_stu_ids) == 1 and len(next_tea_ids) > 1:
        prev_id = tea_label_ids[next_tea_ids[:-1]] 
        prev_token_list = tokenizer_B.convert_ids_to_tokens(prev_id)
        prev_token = "".join(prev_token_list)
        aligned_vec = torch.full_like(back_logit[next_tea_ids[0]], mask_id)
        topk_values, topk_indices = torch.topk(teacher_logits[next_tea_ids[-2]], k=K, dim=-1) 
        token_ids = topk_indices.tolist()
        scores = topk_values.tolist()
        for j in range(K):
            cur_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
            new_token = prev_token + cur_token
            convert_token = re.sub(f"^{teacher_special_token}", student_special_token, new_token)
            new_id = tokenizer_A.convert_tokens_to_ids(convert_token)
            if new_id != tokenizer_A.unk_token_id:
                if aligned_vec[new_id] == mask_id:
                    aligned_vec[new_id] = scores[j]
            else:
                convert_token_ = new_token.replace(teacher_special_token, " ")
                new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                if aligned_vec[new_id[0]] == mask_id:
                    aligned_vec[new_id[0]] = scores[j]
        tokenizer_aligned_teacher_logit = teacher_logits[next_tea_ids[-2]]
    elif len(next_stu_ids) > 1 and len(next_tea_ids) > 1:
        stu_token = tokenizer_A.convert_ids_to_tokens([stu_label_ids[next_stu_ids[0]]])[0]
        tea_token = tokenizer_B.convert_ids_to_tokens([tea_label_ids[next_tea_ids[0]]])
        if tea_idx == 0 and tea_token != teacher_special_token:
            tea_token = re.sub(f"^{teacher_special_token}", "", tea_token[0])
        else:
            tea_token = re.sub(f"^{teacher_special_token}", student_special_token, tea_token[0])
        aligned_vec = torch.full_like(back_logit[tea_idx], mask_id)
        if tea_token.startswith(stu_token):
            if start:
                topk_values, topk_indices = torch.topk(align_idx0[0], k=K, dim=-1)
                tokenizer_aligned_teacher_logit = align_idx0[0]
            else:
                topk_values, topk_indices = torch.topk(teacher_logits[tea_idx], k=K, dim=-1)
                tokenizer_aligned_teacher_logit = teacher_logits[tea_idx]
            token_ids = topk_indices.tolist()
            scores = topk_values.tolist()
            for j in range(K):
                cur_tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
                if start and cur_tea_token != teacher_special_token:
                    convert_token_ = re.sub(f"^{teacher_special_token}", "", cur_tea_token)
                else:
                    convert_token_ = re.sub(f"^{teacher_special_token}", student_special_token, cur_tea_token)
                corres_stu_token_id = tokenizer_A.convert_tokens_to_ids(convert_token_)
                if corres_stu_token_id != tokenizer_A.unk_token_id:
                    if aligned_vec[corres_stu_token_id] == mask_id:
                        aligned_vec[corres_stu_token_id] = scores[j]
                else:
                    convert_token_ = re.sub(f"^{student_special_token}", " ", convert_token_)
                    new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                    if aligned_vec[new_id[0]] == mask_id:
                        aligned_vec[new_id[0]] = scores[j]
        elif stu_token.startswith(tea_token):
            topk_values, topk_indices = torch.topk(teacher_logits[next_tea_ids[0]], k=K, dim=-1)
            token_ids = topk_indices.tolist()
            scores = topk_values.tolist()
            for j in range(K):
                next_tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
                combine_token = tea_token + next_tea_token
                corres_stu_token_id = tokenizer_A.convert_tokens_to_ids(combine_token)
                if corres_stu_token_id != tokenizer_A.unk_token_id:
                    if aligned_vec[corres_stu_token_id] == mask_id:
                        aligned_vec[corres_stu_token_id] = scores[j]
                else:
                    convert_token_ = re.sub(f"^{student_special_token}", " ", combine_token)
                    new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                    if aligned_vec[new_id[0]] == mask_id:
                        aligned_vec[new_id[0]] = scores[j]
            tokenizer_aligned_teacher_logit = teacher_logits[next_tea_ids[0]]
        else:
            aligned_vec = teacher_logits_projected[tea_idx]
            tokenizer_aligned_teacher_logit = teacher_logits[tea_idx]
    elif len(next_stu_ids) > 1 and len(next_tea_ids) == 1:
        aligned_vec = torch.full_like(back_logit[tea_idx], mask_id)
        if start:
            topk_values, topk_indices = torch.topk(align_idx0[0], k=K, dim=-1)
            tokenizer_aligned_teacher_logit = align_idx0[0]
        else:
            topk_values, topk_indices = torch.topk(teacher_logits[tea_idx], k=K, dim=-1)
            tokenizer_aligned_teacher_logit = teacher_logits[tea_idx]
        token_ids = topk_indices.tolist()
        scores = topk_values.tolist()
        for j in range(K):
            cur_tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
            if start and cur_tea_token != teacher_special_token:
                convert_token_ = re.sub(f"^{teacher_special_token}", "", cur_tea_token)
            else:
                convert_token_ = re.sub(f"^{teacher_special_token}", student_special_token, cur_tea_token)
            corres_stu_token_id = tokenizer_A.convert_tokens_to_ids(convert_token_)
            if corres_stu_token_id != tokenizer_A.unk_token_id:
                if aligned_vec[corres_stu_token_id] == mask_id:
                    aligned_vec[corres_stu_token_id] = scores[j]
            else:
                convert_token_ = re.sub(f"^{student_special_token}", " ", convert_token_)
                new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                if aligned_vec[new_id[0]] == mask_id:
                    aligned_vec[new_id[0]] = scores[j]
    else:
        aligned_vec = teacher_logits_projected[tea_idx]
        tokenizer_aligned_teacher_logit = teacher_logits[tea_idx]
    return aligned_vec, tokenizer_aligned_teacher_logit

def sliding_window_align_onlyQ(args,K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, alignment, student_per_step_logit, teacher_logits, teacher_logits_projected, teacher_special_token, student_special_token, align_idx0):
    aligned_logits = []
    tokenizer_aligned_teacher_logits = []
    tot = 0
    tea_idx = 0
    next_stu_ids, next_tea_ids = alignment[0]
    aligned_vec, tokenizer_aligned_teacher_logit = detect_next_token(K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, tea_idx, next_stu_ids, next_tea_ids, student_per_step_logit, \
        teacher_logits, teacher_logits_projected, align_idx0, teacher_special_token, student_special_token, start=True)
    aligned_logits.append(aligned_vec.unsqueeze(0))
    tokenizer_aligned_teacher_logits.append(tokenizer_aligned_teacher_logit.unsqueeze(0))
    back_logit = student_per_step_logit if student_per_step_logit.size(0) > teacher_logits_projected.size(0) else teacher_logits_projected
    for g in range(len(alignment)):
        stu_ids, tea_ids = alignment[g]
        if g+1 < len(alignment):
            next_stu_ids, next_tea_ids = alignment[g+1]
        else:
            next_stu_ids, next_tea_ids = [], []
        if g-1 > 0:
            prev_stu_ids, prev_tea_ids = alignment[g-1]
        else:
            prev_tea_ids = None

        if len(stu_ids) == 1 and len(tea_ids) == 1:
            tea_idx = tea_ids[0]
        elif len(stu_ids) == 1 and len(tea_ids) > 1:
            tea_idx = tea_ids[-1]
        elif len(stu_ids) > 1:
            tea_idx = tea_ids[-1]
            prev_aligned_vec = torch.full_like(back_logit[stu_ids[:-1]], mask_id)
            if len(tea_ids) > 1:
                prev_token_list = tokenizer_B.convert_ids_to_tokens(tea_label_ids[tea_ids[:-1]])
                prev_token = "".join(prev_token_list)
                if tea_ids[0] == 0 and prev_token != teacher_special_token:
                    prev_token = re.sub(f"^{teacher_special_token}", "", prev_token)
                else:
                    prev_token = re.sub(f"^{teacher_special_token}", student_special_token, prev_token)
                topk_values, topk_indices = torch.topk(teacher_logits[tea_ids[-2]], k=K, dim=-1)
                token_ids = topk_indices.tolist()
                scores = topk_values.tolist()
                for j in range(K):
                    cur_tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
                    cur_tea_token = re.sub(f"^{teacher_special_token}", student_special_token, cur_tea_token)
                    combine_token = prev_token + cur_tea_token
                    corres_stu_token_id = tokenizer_A.convert_tokens_to_ids(combine_token)
                    if corres_stu_token_id != tokenizer_A.unk_token_id:
                        if prev_aligned_vec[0,corres_stu_token_id] == mask_id:
                            prev_aligned_vec[0,corres_stu_token_id] = scores[j]
                    else:
                        convert_token_ = re.sub(f"^{student_special_token}", " ", combine_token)
                        new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                        if len(new_id) == 1:
                            if prev_aligned_vec[0,corres_stu_token_id] == mask_id:
                                prev_aligned_vec[0,corres_stu_token_id] = scores[j]
                        else:
                            for i in range(len(stu_ids)-1):
                                if (i+1) < len(new_id):
                                    if prev_aligned_vec[i, new_id[i+1]] == mask_id:
                                        prev_aligned_vec[i, new_id[i+1]] = scores[j]
                tokenizer_aligned_teacher_logits += [teacher_logits[tea_ids[-2]].unsqueeze(0)] * (len(stu_ids) - 1)
            elif len(tea_ids) == 1:
                prev_token = ""
                if prev_tea_ids:
                    topk_values, topk_indices = torch.topk(teacher_logits[prev_tea_ids[-1]], k=K, dim=-1)
                    tokenizer_aligned_teacher_logits += [teacher_logits[prev_tea_ids[-1]].unsqueeze(0)] * (len(stu_ids) - 1)
                else:
                    topk_values, topk_indices = torch.topk(align_idx0[0], k=K, dim=-1)
                    tokenizer_aligned_teacher_logits += [align_idx0[0].unsqueeze(0)] * (len(stu_ids) - 1)
                token_ids = topk_indices.tolist()
                scores = topk_values.tolist()
                for j in range(K):
                    cur_tea_token = tokenizer_B.convert_ids_to_tokens(token_ids[j])
                    if tea_ids[0] == 0 and cur_tea_token != teacher_special_token:
                        cur_tea_token = re.sub(f"^{teacher_special_token}", "", cur_tea_token)
                    else:
                        cur_tea_token = re.sub(f"^{teacher_special_token}", student_special_token, cur_tea_token)
                    combine_token = prev_token + cur_tea_token
                    convert_token_ = re.sub(f"^{student_special_token}", " ", combine_token)
                    new_id = tokenizer_A.encode(convert_token_, add_special_tokens=False)
                    if len(new_id) > 1:
                        for i in range(len(stu_ids)-1):
                            if (i+1) < len(new_id):
                                if prev_aligned_vec[i, new_id[i+1]] == mask_id:
                                    prev_aligned_vec[i, new_id[i+1]] = scores[j]
            aligned_logits.append(prev_aligned_vec)
        aligned_vec, tokenizer_aligned_teacher_logit = detect_next_token(K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, tea_idx, next_stu_ids, next_tea_ids, student_per_step_logit, \
            teacher_logits, teacher_logits_projected, align_idx0, teacher_special_token, student_special_token, start=False)            
        aligned_logits.append(aligned_vec.unsqueeze(0))
        tokenizer_aligned_teacher_logits.append(tokenizer_aligned_teacher_logit.unsqueeze(0))
    aligned_output = torch.cat(aligned_logits, dim=0)
    tokenizer_aligned_output = torch.cat(tokenizer_aligned_teacher_logits, dim=0)
    return aligned_output, tokenizer_aligned_output


class SEDILogitDistillation(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.K = args.K
        self.padding_id = padding_id
        self.args = args
    
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
        self.distiller = distiller
        
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
                    
        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )
            
        log = {}
        
        Q = distiller.Q
        kd_loss, log = self.compute_sedi_distillation_loss(
            Q, outputs, teacher_outputs, input_data, output_data, distiller, log
        )
        
        log["kd_loss"] = kd_loss
        
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]
        log["nll_loss"] = ce_loss
        
        loss = (1.0 - self.kd_rate) * ce_loss + self.kd_rate * kd_loss
        log["loss"] = loss

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        
        return loss / batch_denom, logging_output
    
    def compute_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs))
        inf_mask = logits.isinf()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["forward_kl"] = kld

        return kld


    def compute_sedi_distillation_loss(self, Q, outputs, teacher_outputs, inputs, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits 
        student_tokenizer = distiller.student_tokenizer
        teacher_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]
        
        student_special_token = specialtokens[self.args.model_type]
        teacher_special_token = specialtokens[self.args.teacher_model_type]
        
        mask_id = -10000

        bsz = target.shape[0]
        aligned_tea_logits = []
        entropy_loss = 0
        for i in range(bsz):
            assert self.padding_id in target[i]
            stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
            stu_input_ids = inputs["input_ids"][i, stu_content_idx]
            text = student_tokenizer.decode(stu_input_ids[1:], skip_special_tokens=True)
            tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
            tea_input_ids = inputs[f"teacher_{distiller.teacher_model_type}_input_ids"][i, tea_content_idx]
            tea_per_step_logits = teacher_outputs.logits[i, tea_content_idx, :] 
            tea_text = teacher_tokenizer.decode(tea_input_ids[1:], skip_special_tokens=True)
            if len(text) > len(tea_text):
                text = text[:len(tea_text)]
            if len(tea_text) > len(text):
                text = text[:len(text)]
            
            alignment, student_id_len, teacher_id_len = align_tokenizers(text, student_tokenizer, teacher_tokenizer, stu_input_ids[1:], tea_input_ids[1:])
            merged = merge_alignments_simple(alignment)
            
            end_student_len = merged[-1][0][-1]+1
            end_teacher_len = merged[-1][1][-1]+1
            if end_student_len != stu_input_ids.size(0):
                if end_student_len < stu_input_ids.size(0):
                    stu_input_ids = stu_input_ids[:end_student_len+1]
                    tea_input_ids = tea_input_ids[:end_teacher_len+1]
            teacher_logits_projected = torch.matmul(tea_per_step_logits.to(Q.dtype), Q)  
            student_per_step_logit = student_logits[i, stu_content_idx[:end_student_len+1], :].to(Q.dtype)
            
            aligned, tokenizer_aligned_output = sliding_window_align_onlyQ(self.args,self.K, mask_id, student_tokenizer, teacher_tokenizer, stu_input_ids[1:], tea_input_ids[1:], merged, student_per_step_logit, \
                tea_per_step_logits[1:].to(Q.dtype), teacher_logits_projected[1:].to(Q.dtype), teacher_special_token, student_special_token, tea_per_step_logits[0,:].unsqueeze(0))
            assert tokenizer_aligned_output.size(0) == aligned.size(0)
            next_token_ids_post = torch.argmax(aligned, dim=-1)
            tag = torch.ones_like(next_token_ids_post[:-1], dtype=torch.bool).to(next_token_ids_post.device)
            mismatch = next_token_ids_post[:-1] != stu_input_ids[1:]
            tag[mismatch] = False 
            one_hot_logits = F.one_hot(stu_input_ids[1:], num_classes=student_logits.shape[-1])
            one_hot_logits = (1 - one_hot_logits) * (-100000) + (one_hot_logits) * 100
            aligned[:-1][~tag] = one_hot_logits[~tag].to(aligned.dtype)
            transformed_stu_logit = student_per_step_logit * distiller.entropy_adapter.to(dtype=student_per_step_logit.dtype)
            aligned = aligned + transformed_stu_logit
            
            def entropy(logits):
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                return -torch.sum(probs * log_probs, dim=-1)  # (s,)
            
            entropy_teacher = entropy(tokenizer_aligned_output.detach())
            entropy_aligned = entropy(aligned)
            entropy_align_loss = torch.mean((entropy_aligned - entropy_teacher)**2)
            entropy_loss += entropy_align_loss
            
            aligned_tea_per_step_logits = student_logits[i].float().detach()
            aligned_tea_per_step_logits[stu_content_idx[:student_id_len+1]] = aligned.float() 
            aligned_tea_logits.append(aligned_tea_per_step_logits)
            
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)

        kl_loss = self.compute_forward_kl_divergence(
            student_logits, 
            aligned_tea_logits, 
            output_data["label"],
            log=log
        )
        log["kl_loss"] = kl_loss + entropy_loss
        return kl_loss+entropy_loss, log

    