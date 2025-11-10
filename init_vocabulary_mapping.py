from transformers import GPT2Tokenizer,AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json
import transformers
import torch
from tqdm import tqdm
from collections import defaultdict
import unicodedata
import string
import argparse
import re
import os

specialtokens = {
    "gpt2": "Ġ", 
    "llama": "▁",
    "qwen": "Ġ",
    "deepseek": "Ġ",
    "pythia": "Ġ",
    "opt": "Ġ",
    }


def find_best_Q(stu_token_map, tea_token_map, lensA, lensB, special_stu_id, special_tea_id, student_tokenizer, teacher_tokenizer):
    match_matrix = torch.zeros((lensB,lensA), dtype=torch.bfloat16)
    match_matrix[special_tea_id,special_stu_id] = 1.0
    print(match_matrix.shape)
    special_tea_token = teacher_tokenizer.convert_ids_to_tokens(special_tea_id)
    special_stu_token = student_tokenizer.convert_ids_to_tokens(special_stu_id)
    for name, token in student_tokenizer.special_tokens_map.items():
        print(f"{name}: '{token}' -> id: {student_tokenizer.convert_tokens_to_ids(token)}")
        for name2, token2 in teacher_tokenizer.special_tokens_map.items():
            if name2 == name:
                if isinstance(token2, list) or name2 == "additional_special_tokens":
                    continue
                else:
                    print(f"{name2}: '{token2}' -> id: {teacher_tokenizer.convert_tokens_to_ids(token2)}")
                    match_matrix[teacher_tokenizer.convert_tokens_to_ids(token2),student_tokenizer.convert_tokens_to_ids(token)] = 1.0
    
    len_to_tokens_stu = defaultdict(list)
    for j, id in enumerate(stu_token_map):
        token = stu_token_map[id]
        len_token = len(token)
        len_to_tokens_stu[len_token].append((token, id))
    
    for i, id in tqdm(enumerate(tea_token_map), total=len(tea_token_map), desc="Processing tea_token_map"):
        a_token = tea_token_map[id]
        a_len = len(a_token)
        if a_token != special_tea_token and a_token not in teacher_tokenizer.special_tokens_map.values():
            for b_len in len_to_tokens_stu:
                if b_len == a_len:
                    for b_token, j in len_to_tokens_stu[b_len]:
                        if a_token == b_token:
                            match_matrix[i, j] = 1.0
    
    return match_matrix

def find_best_mapping(stu_token_map, tea_token_map, lensA, lensB, special_stu_id, special_tea_id, student_tokenizer, teacher_tokenizer):
    tea_to_stu_id_mapping = dict()
    token_mapping = {}
    tea_to_stu_id_mapping[special_tea_id] = special_stu_id
    token_mapping[tea_token_map[special_tea_id]] = student_tokenizer.convert_ids_to_tokens(special_stu_id)
    special_tea_token = teacher_tokenizer.convert_ids_to_tokens(special_tea_id)
    special_stu_token = student_tokenizer.convert_ids_to_tokens(special_stu_id)
    for name, token in student_tokenizer.special_tokens_map.items():
        print(f"{name}: '{token}' -> id: {student_tokenizer.convert_tokens_to_ids(token)}")
        for name2, token2 in teacher_tokenizer.special_tokens_map.items():
            if name2 == name:
                print(f"{name2}: '{token2}' -> id: {teacher_tokenizer.convert_tokens_to_ids(token2)}")
                tea_to_stu_id_mapping[teacher_tokenizer.convert_tokens_to_ids(token2)] = student_tokenizer.convert_tokens_to_ids(token)
                token_mapping[token2] = token
    
    len_to_tokens_stu = defaultdict(list)
    for j, id in enumerate(stu_token_map):
        token = stu_token_map[id]
        len_token = len(token)
        len_to_tokens_stu[len_token].append((token, id))
    
    for i, id in tqdm(enumerate(tea_token_map), total=len(tea_token_map), desc="Processing tea_token_map"):
        a_token = tea_token_map[id]
        a_len = len(a_token)
        if a_token != special_tea_token and a_token not in teacher_tokenizer.special_tokens_map.values():
            for b_len in len_to_tokens_stu:
                if b_len == a_len:
                    for b_token, j in len_to_tokens_stu[b_len]:
                        if a_token == b_token:
                            tea_to_stu_id_mapping[i] = j
                            token_mapping[a_token] = b_token
    print("exact match teacher to student: ",len(tea_to_stu_id_mapping))
    
    return tea_to_stu_id_mapping, token_mapping
    

def is_byte_token(token: str) -> bool:
    if token.startswith("<0x") and token.endswith(">"):
        return True
    if len(token) == 1 and 0 <= ord(token) <= 255:
        return True
    return False


def init_mapping(args):
    student_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    
    base_tokenizer_json_path = os.path.join(args.model_path, "tokenizer.json")
    base_vocab_json_path = os.path.join(args.model_path, "vocab.json")
    if os.path.exists(base_tokenizer_json_path):
        with open(base_tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            stu_vocab_dict = data["model"]["vocab"]
        stu_id_to_token = {v: k for k, v in stu_vocab_dict.items()}
        if "added_tokens" in data:
            for token_info in data["added_tokens"]:
                token_id = token_info["id"]
                token_str = token_info["content"]
                stu_id_to_token[token_id] = token_str
    elif os.path.exists(base_vocab_json_path):
        with open(base_vocab_json_path, "r", encoding="utf-8") as f:
            stu_vocab_dict = json.load(f)
        stu_id_to_token = {v: k for k, v in stu_vocab_dict.items()}
    else:
        raise FileNotFoundError(
            f"Neither 'tokenizer.json' nor 'vocab.json' found in {args.base_model_name_or_path}"
        )

    blending_tokenizer_json_path = os.path.join(args.teacher_model_path, "tokenizer.json")
    blending_vocab_json_path = os.path.join(args.teacher_model_path, "vocab.json")
    if os.path.exists(blending_tokenizer_json_path):      
        with open(blending_tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            tea_vocab_dict = data["model"]["vocab"]
        tea_id_to_token = {v: k for k, v in tea_vocab_dict.items()}
        if "added_tokens" in data:
            for token_info in data["added_tokens"]:
                token_id = token_info["id"]
                token_str = token_info["content"]
                tea_id_to_token[token_id] = token_str
    elif os.path.exists(blending_vocab_json_path):
        with open(blending_vocab_json_path, "r", encoding="utf-8") as f:
            tea_vocab_dict = json.load(f)
        tea_id_to_token = {v: k for k, v in tea_vocab_dict.items()}
    else:
        raise FileNotFoundError(
            f"Neither 'tokenizer.json' nor 'vocab.json' found in {args.base_model_name_or_path}"
        )
    
    base_model_special_token = specialtokens[args.model_type]
    blending_model_special_token = specialtokens[args.teacher_model_type]
    print("base_model_special_token: ", base_model_special_token)
    print("blending_model_special_token: ", blending_model_special_token)
    
    special_stu_id = student_tokenizer.convert_tokens_to_ids(base_model_special_token)
    special_tea_id = teacher_tokenizer.convert_tokens_to_ids(blending_model_special_token)
    special_stu_token = specialtokens[args.model_type]
    special_tea_token = specialtokens[args.teacher_model_type]
    print("special ids:", special_stu_id, special_tea_id)

    tea_token_map = {}
    for i in range(len(tea_id_to_token)):
        token = tea_id_to_token[i]
        decoded = teacher_tokenizer.decode([i], skip_special_tokens=False)
        if decoded in ['\n', '\t', '\\', '/', '▁', 'Ġ', '▁▁', '▁▁▁', '▁▁▁▁', '\r']:
            token = decoded
        if i < 256:
            if decoded != '�':
                token = decoded
        if token == '':
            token = " "
        if token.startswith(special_tea_token) and token!= special_tea_token:
            if set(token) != {special_tea_token}:
                token = re.sub(f"^{special_tea_token}", special_stu_token, token)
            else:
                token = decoded
        tea_token_map[i] = token
    
    stu_token_map = {}
    for i in range(len(stu_id_to_token)):
        token = stu_id_to_token[i]
        decoded = student_tokenizer.decode([i], skip_special_tokens=False)
        if i < 256:
            if decoded != '�':
                token = decoded
        if decoded in ['\n', '\t', '\\', '/', '▁', 'Ġ', '▁▁', '▁▁▁', '▁▁▁▁', '\r','.\n']:
            token = decoded
        stu_token_map[i] = token

    lensA = len(stu_token_map)
    lensB = len(tea_token_map)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    student_model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config)
    if hasattr(student_model, "lm_head"):
        output_weight = student_model.lm_head.weight
    elif hasattr(student_model, "embed_out"):
        output_weight = student_model.embed_out.weight
    else:
        raise NotImplementedError("Cannot find output embedding weight in the model.")
    if output_weight.size(0) >= lensA:
        lensA = output_weight.size(0)
    config = AutoConfig.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, config=config)
    if hasattr(teacher_model, "lm_head"):
        output_weight = teacher_model.lm_head.weight
    elif hasattr(teacher_model, "embed_out"):
        output_weight = teacher_model.embed_out.weight
    else:
        raise NotImplementedError("Cannot find output embedding weight in the model.")
    if output_weight.size(0) >= lensB:
        lensB = output_weight.size(0)
    print("student vocab size: ", lensA)
    print("teacher vocab size: ", lensB)
    
    Q = find_best_Q(stu_token_map, tea_token_map, lensA, lensB, special_stu_id, special_tea_id, student_tokenizer, teacher_tokenizer)
    return Q
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="uld")
    parser.add_argument("--model_type", type=str, default="opt")
    parser.add_argument("--teacher_model_type", type=str, default="pythia")
    parser.add_argument("--model_path", type=str, default="opt-125m")    
    parser.add_argument("--teacher_model_path", type=str, default="dolly-pythia-v2-3b")    
    parser.add_argument("--save_path", type=str, default="./")
    args, rest = parser.parse_known_args()

    Q = init_mapping(args)