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

def align_tokenizers(
    text: str,
    student_tokenizer: PreTrainedTokenizer,
    teacher_tokenizer: PreTrainedTokenizer,
    student_ids: List[int],
    teacher_ids: List[int],
):
    student_spans, stu_ids_ = get_token_spans(text, student_ids, student_tokenizer)
    teacher_spans, tea_ids_ = get_token_spans(text, teacher_ids, teacher_tokenizer)

    if len(student_spans) == 0 or len(teacher_spans) == 0:
        return [], len(stu_ids_), len(tea_ids_)

    end_s = student_spans[-1][1]
    end_t = teacher_spans[-1][1]
    end = min(end_s, end_t)

    stu_valid = [i for i, (_, b) in enumerate(student_spans) if b <= end]
    tea_valid = [i for i, (_, b) in enumerate(teacher_spans) if b <= end]

    if len(stu_valid) == 0 or len(tea_valid) == 0:
        return [], len(stu_ids_), len(tea_ids_)

    def span_overlaps(s_start, s_end, t_start, t_end):
        if t_start == t_end:
            return s_start <= t_start < s_end
        if s_start == s_end:
            return t_start <= s_start < t_end
        return not (t_end <= s_start or t_start >= s_end)

    alignment = []
    for i in stu_valid:
        s_start, s_end = student_spans[i]
        matching_teacher = []
        for k in tea_valid:
            t_start, t_end = teacher_spans[k]
            if span_overlaps(s_start, s_end, t_start, t_end):
                matching_teacher.append(k)
        if matching_teacher:
            alignment.append((i, matching_teacher))

    if len(alignment) == 0:
        return [], len(stu_ids_), len(tea_ids_)

    return alignment, len(stu_ids_), len(tea_ids_)



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

def longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def group_rerank_builder(args, K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids,
    stu_ids, tea_ids, group_index, full_alignment, student_per_step_logit, teacher_logits,
    teacher_logits_projected, teacher_special_token, student_special_token, align_idx0, student_model, sft_per_step_logit, stu_context_ids):

    group_aligned_vecs = []
    group_teacher_logits = []

    device = teacher_logits.device
    group_len = len(stu_ids)

    gt_teacher_ids = [int(tea_label_ids[idx]) for idx in tea_ids]
    gt_teacher_tokens = tokenizer_B.convert_ids_to_tokens(gt_teacher_ids)

    gt_student_ids = [int(stu_label_ids[idx]) for idx in stu_ids]
    gt_student_tokens = tokenizer_A.convert_ids_to_tokens(gt_student_ids)
    
    # print("Ground truth student group tokens:", gt_student_tokens)
    # print("Ground truth teacher group tokens:", gt_teacher_tokens)

    group_tokens_info = []

    # 防止 K 超过词表大小
    K = min(K, sft_per_step_logit.size(-1))

    for pos in range(group_len):
        fixed_student_prefix_ids = list(stu_context_ids) + [int(stu_label_ids[idx]) for idx in stu_ids[:pos]]

        cur_seq_pos = len(stu_context_ids) + pos
        topk_vals, topk_ids = torch.topk(sft_per_step_logit[cur_seq_pos], k=K, dim=-1)

        valid_candidates = [] 

        for cand_id in topk_ids.tolist():
            cur_student_ids = fixed_student_prefix_ids + [cand_id]

            for _ in range(group_len - pos - 1):
                input_ids = torch.tensor(cur_student_ids, dtype=torch.long, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)

                with torch.no_grad():
                    outputs = student_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    next_logits = outputs.logits[0, -1, :]
                    next_id = torch.argmax(next_logits, dim=-1).item()

                cur_student_ids.append(next_id)

            group_student_ids = cur_student_ids[len(stu_context_ids):]
            group_text = tokenizer_A.decode(group_student_ids, skip_special_tokens=True)
            # print("Student rollout group tokens:", group_text)
            group_teacher_ids = tokenizer_B.encode(group_text, add_special_tokens=False)

            if len(group_teacher_ids) == 0:
                continue

            common_len = longest_common_prefix_len(gt_teacher_ids, group_teacher_ids)

            if common_len >= len(group_teacher_ids):
                mapped_pos = len(group_teacher_ids) - 1
            else:
                mapped_pos = common_len

            # 这里 mapped_pos 是 group_teacher_ids 的局部位置
            if mapped_pos >= len(group_teacher_ids):
                mapped_pos = len(group_teacher_ids) - 1

            mapped_tid = group_teacher_ids[mapped_pos]

            if mapped_tid >= teacher_logits.size(-1):
                continue

            # teacher 全局位置，要加 tea_ids[0]，并且要防越界
            teacher_abs_pos = min(tea_ids[0] + mapped_pos - 1, teacher_logits.size(0) - 1)

            score = teacher_logits[teacher_abs_pos][mapped_tid]
            valid_candidates.append((cand_id, score, mapped_pos))
        
        #     topk_teacher_vals, topk_teacher_ids = torch.topk(teacher_logits[teacher_abs_pos], k=K, dim=-1)
        #     teacher_topk_tokens = tokenizer_B.convert_ids_to_tokens(topk_teacher_ids.tolist())
        #     teacher_topk_logits = topk_teacher_vals.tolist()
        #     teacher_abs_pos_for_print = teacher_abs_pos

        # group_tokens_info.append({
        #     "pos": pos,
        #     "sft_topk_tokens": [tokenizer_A.convert_ids_to_tokens(tid) for tid in topk_ids.tolist()],
        #     "sft_topk_logits": topk_vals.tolist(),
        #     "student_rollout_tokens": tokenizer_A.convert_ids_to_tokens(group_student_ids),
        #     "teacher_mapped_tokens": tokenizer_B.convert_ids_to_tokens(group_teacher_ids) if len(group_teacher_ids) > 0 else [],
        #     "teacher_abs_pos": teacher_abs_pos_for_print,
        #     "teacher_topk_tokens": teacher_topk_tokens,
        #     "teacher_topk_logits": teacher_topk_logits,
        # })


        aligned_vec = torch.full_like(student_per_step_logit[0], float(mask_id))

        # 不管 valid_candidates 是否为空，fallback 都应该是 teacher 全局位置
        fallback_pos = min(tea_ids[0] + pos, teacher_logits.size(0) - 1)
        tokenizer_aligned_teacher_logit = teacher_logits[fallback_pos]

        if len(valid_candidates) > 0:
            for cand_id, score, _ in valid_candidates:
                aligned_vec[cand_id] = score

        group_aligned_vecs.append(aligned_vec.unsqueeze(0))
        group_teacher_logits.append(tokenizer_aligned_teacher_logit.unsqueeze(0))

    group_aligned_vecs = torch.cat(group_aligned_vecs, dim=0)
    group_teacher_logits = torch.cat(group_teacher_logits, dim=0)

    # print("\n=== Group Tokens Info ===")
    # for info in group_tokens_info:
    #     print(f"Position {info['pos']}:")
    #     print("  SFT top-k tokens:", info["sft_topk_tokens"])
    #     print("  SFT top-k logits:", [round(v, 4) for v in info["sft_topk_logits"]])
    #     print("  Student rollout tokens:", info["student_rollout_tokens"])
    #     print("  Teacher mapped tokens:", info["teacher_mapped_tokens"])
    #     print("  Teacher abs pos:", info["teacher_abs_pos"])
    #     print("  Teacher top-k tokens:", info["teacher_topk_tokens"])
    #     print("  Teacher top-k logits:", [round(v, 4) for v in info["teacher_topk_logits"]])
    #     print("-" * 40)

    return group_aligned_vecs, group_teacher_logits


def sliding_window_align_onlyQ(args,K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, alignment, student_per_step_logit, teacher_logits, teacher_logits_projected, teacher_special_token, student_special_token, align_idx0, student_model, sft_per_step_logit):
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
            aligned_vec, tokenizer_aligned_teacher_logit = detect_next_token(K, mask_id, tokenizer_A, tokenizer_B, stu_label_ids, tea_label_ids, tea_idx, next_stu_ids, next_tea_ids, student_per_step_logit, \
                teacher_logits, teacher_logits_projected, align_idx0, teacher_special_token, student_special_token, start=False)            
            aligned_logits.append(aligned_vec.unsqueeze(0))
            tokenizer_aligned_teacher_logits.append(tokenizer_aligned_teacher_logit.unsqueeze(0))
            continue
        

        stu_context_ids = []
        for prev_g in range(g):
            prev_stu_ids, _ = alignment[prev_g]
            stu_context_ids.extend([int(stu_label_ids[idx]) for idx in prev_stu_ids])

        group_student_logits, group_pseudo_logits = group_rerank_builder(
            args=args,
            K=40,
            mask_id=mask_id,
            tokenizer_A=tokenizer_A,  
            tokenizer_B=tokenizer_B,   
            stu_label_ids=stu_label_ids,
            tea_label_ids=tea_label_ids,
            stu_ids=stu_ids,
            tea_ids=tea_ids,
            group_index=g,
            full_alignment=alignment,
            student_per_step_logit=student_per_step_logit,
            teacher_logits=teacher_logits,
            teacher_logits_projected=teacher_logits_projected,
            teacher_special_token=teacher_special_token,
            student_special_token=student_special_token,
            align_idx0=align_idx0,
            student_model=student_model,
            sft_per_step_logit=sft_per_step_logit, 
            stu_context_ids=stu_context_ids,
        )

        if group_student_logits.dim() == 1:
            group_student_logits = group_student_logits.unsqueeze(0)
        if group_pseudo_logits.dim() == 1:
            group_pseudo_logits = group_pseudo_logits.unsqueeze(0)

        aligned_logits.append(group_student_logits)
        tokenizer_aligned_teacher_logits.append(group_pseudo_logits)

    aligned_output = torch.cat(aligned_logits, dim=0)
    tokenizer_aligned_output = torch.cat(tokenizer_aligned_teacher_logits, dim=0)
    return aligned_output, tokenizer_aligned_output


class SFTSEDILogitDistillation(CrossEntropyLoss):
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
        student_model_for_sft = distiller.student_model_for_sft
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

        with torch.no_grad():
            sft_outputs = student_model_for_sft(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                output_hidden_states=True
            )
        
        log = {}
        
        Q = distiller.Q
        kd_loss, log = self.compute_sedi_distillation_loss(
            Q, outputs, teacher_outputs, sft_outputs, input_data, output_data, distiller, log
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

    def debug_print_merged_groups(
        self,
        text: str,
        student_tokenizer: PreTrainedTokenizer,
        teacher_tokenizer: PreTrainedTokenizer,
        student_ids: List[int],
        teacher_ids: List[int],
        merged,  # List[Tuple[List[int], List[int]]]
        max_groups: int = 200,
    ):
        stu_spans, _ = get_token_spans(text, student_ids, student_tokenizer)
        tea_spans, _ = get_token_spans(text, teacher_ids, teacher_tokenizer)

        stu_tokens = student_tokenizer.convert_ids_to_tokens([int(x) for x in student_ids])
        tea_tokens = teacher_tokenizer.convert_ids_to_tokens([int(x) for x in teacher_ids])

        print("\n================= ALIGNMENT (merged groups) =================")
        print(f"num_groups={len(merged)}")
        print("-------------------------------------------------------------")

        for gi, (s_group, t_group) in enumerate(merged[:max_groups]):
            s_toks = [stu_tokens[i] for i in s_group]
            t_toks = [tea_tokens[i] for i in t_group]

            s_span = (stu_spans[s_group[0]][0], stu_spans[s_group[-1]][1]) if s_group else None
            t_span = (tea_spans[t_group[0]][0], tea_spans[t_group[-1]][1]) if t_group else None

            s_str = text[s_span[0]:s_span[1]] if s_span else ""
            t_str = text[t_span[0]:t_span[1]] if t_span else ""

            print(f"[G{gi}]")
            print(f"  S idx={s_group} toks={s_toks} span={s_span} str={repr(s_str)}")
            print(f"  T idx={t_group} toks={t_toks} span={t_span} str={repr(t_str)}")
            print("")

        print("=============================================================\n")

    def compute_sedi_distillation_loss(self, Q, outputs, teacher_outputs, sft_outputs, inputs, output_data, distiller, log
    ):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits 
        sft_logits = sft_outputs.logits 
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
            if alignment is None or len(alignment) == 0:
                continue
            merged = merge_alignments_simple(alignment)
            if merged is None or len(merged) == 0:
                continue

            # self.debug_print_merged_groups(
            #     text,
            #     student_tokenizer,
            #     teacher_tokenizer,
            #     student_ids=stu_input_ids[1:].tolist(),
            #     teacher_ids=tea_input_ids[1:].tolist(),
            #     merged=merged,
            # )
            
            end_student_len = merged[-1][0][-1]+1
            end_teacher_len = merged[-1][1][-1]+1
            if end_student_len != stu_input_ids.size(0):
                if end_student_len < stu_input_ids.size(0):
                    stu_input_ids = stu_input_ids[:end_student_len+1]
                    tea_input_ids = tea_input_ids[:end_teacher_len+1]
            teacher_logits_projected = torch.matmul(tea_per_step_logits.to(Q.dtype), Q)  
            student_per_step_logit = student_logits[i, stu_content_idx[:end_student_len+1], :].to(Q.dtype)
            sft_per_step_logit = sft_logits[i, stu_content_idx[:end_student_len+1], :].to(Q.dtype)
            
            aligned, tokenizer_aligned_output = sliding_window_align_onlyQ(self.args,self.K, mask_id, student_tokenizer, teacher_tokenizer, stu_input_ids[1:], tea_input_ids[1:], merged, student_per_step_logit, \
                tea_per_step_logits[1:].to(Q.dtype), teacher_logits_projected[1:].to(Q.dtype), teacher_special_token, student_special_token, tea_per_step_logits[0,:].unsqueeze(0), distiller.student_model, sft_per_step_logit)
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
            L = aligned.shape[0]

            if L > stu_content_idx.shape[0]:
                L = stu_content_idx.shape[0]

            aligned_tea_per_step_logits[
                stu_content_idx[:L]
            ] = aligned.float()

            aligned_tea_logits.append(aligned_tea_per_step_logits)
            
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)

        kl_loss = self.compute_forward_kl_divergence(
            student_logits, 
            aligned_tea_logits, 
            output_data["label"],
            log=log
        )
        log["kl_loss"] = kl_loss 
        # + entropy_loss
        return kl_loss, log

    