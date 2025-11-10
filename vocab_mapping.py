"""Mapping vocabs from different models."""

import argparse
import json
import multiprocessing

import editdistance
import numpy as np
import tqdm
from datasets import DatasetDict, Features, load_dataset, load_from_disk
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import logging
import sys
import os

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = get_logger(__name__)

TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer:"Ġ",
    transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: "Ġ"
}

def get_tokenizer(model_name_or_path, cache_dir, model_max_length):
    if "pythia" in model_name_or_path:
        use_fast = True
    else:
        use_fast = False
    
    kwargs = {
        "use_fast": use_fast,
        "tokenizer_trust_remote_code": False,
        "model_trust_remote_code": False,
    }
    if "llama" in model_name_or_path.lower():
        kwargs["use_fast"] = False
        kwargs["tokenizer_trust_remote_code"] = False
        kwargs["model_trust_remote_code"] = False
    elif "mpt" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    else:
        pass
    logger.info("Loading tokenizer.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=kwargs["use_fast"],
        trust_remote_code=kwargs["tokenizer_trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError
    logger.info(
        f"bos_token: {tokenizer.bos_token}, {tokenizer.bos_token_id} "
        f"eos_token: {tokenizer.eos_token}, {tokenizer.eos_token_id} "
        f"unk_token: {tokenizer.unk_token}, {tokenizer.unk_token_id} "
        f"pad_token: {tokenizer.pad_token}, {tokenizer.pad_token_id} "
    )
    return tokenizer, kwargs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mapping vocabs from different pretrain language models."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="pythia-410m",
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the base model.",
    )
    parser.add_argument(
        "--blending_model_name_or_path",
        type=str,
        default="DeepSeek-R1-Distill-Llama-8B-2",
        help="Path to pretrained model or model identifier from huggingface.co/models. It is the blending model.",
    )
    parser.add_argument(
        "--vocab_mapping_save_dir",
        type=str,
        default="data/vocab_alignment/deepseek_to_pythia/deepseek_pythia_token_exact.json",
        help="The local dir to save processed data.",
    )
    parser.add_argument(
        "--id_mapping_save_dir",
        type=str,
        default="data/vocab_alignment/deepseek_to_pythia/deepseek_pythia_id_exact.json",
        help="The local dir to save processed data.",
    )
    parser.add_argument("--cache_dir", type=str, 
                        default="cache/", help="The cache dir.")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--mode", type=str, default="exact_match" 
    )
    args = parser.parse_args()
    return args

def find_best_mapping(
        x,
        base_tokens,
        blending_model_special_token,
        base_model_special_token,
        best_one=True,
        mode=None,
    ):  
        tmp_x = x.replace(blending_model_special_token, base_model_special_token)
        if tmp_x == base_tokens:
            return x, tmp_x
        else:
            if mode == "exact_match":
                return x, ""
            if best_one:
                return x, min(
                    [(y, editdistance.eval(tmp_x, y)) for y in base_tokens],
                    key=lambda d: d[1],
                )[0]
            else:
                token_and_distance = [
                    (y, editdistance.eval(tmp_x, y)) for y in base_tokens
                ]
                min_distance = min(item[1] for item in token_and_distance)
                shortest_distance_tokens = [
                    item[0] for item in token_and_distance if item[1] == min_distance
                ]
                return x, shortest_distance_tokens

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")

    base_tokenizer, _ = get_tokenizer(
        args.base_model_name_or_path, args.cache_dir, args.model_max_length
    )
    blending_tokenizer, _ = get_tokenizer(
        args.blending_model_name_or_path, args.cache_dir, args.model_max_length
    )
    with open(f"{args.base_model_name_or_path}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        base_lens = config["vocab_size"]
    with open(f"{args.blending_model_name_or_path}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        blending_lens = config["vocab_size"]
    
    base_tokenizer_json_path = os.path.join(args.base_model_name_or_path, "tokenizer.json")
    base_vocab_json_path = os.path.join(args.base_model_name_or_path, "vocab.json")
    if os.path.exists(base_tokenizer_json_path):      
        with open(base_tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            base_vocab = data["model"]["vocab"]
        if "added_tokens" in data:
            for token_info in data["added_tokens"]:
                token_id = token_info["id"]
                token_str = token_info["content"]
                base_vocab[token_str] = token_id
        base_tokens = list(base_vocab.keys())
    elif os.path.exists(base_vocab_json_path):
        with open(base_vocab_json_path, "r", encoding="utf-8") as f:
            base_vocab = json.load(f)
        base_tokens = list(base_vocab.keys())
    else:
        raise FileNotFoundError(
            f"Neither 'tokenizer.json' nor 'vocab.json' found in {args.base_model_name_or_path}"
        )

    blending_tokenizer_json_path = os.path.join(args.blending_model_name_or_path, "tokenizer.json")
    blending_vocab_json_path = os.path.join(args.blending_model_name_or_path, "vocab.json")
    if os.path.exists(blending_tokenizer_json_path):      
        with open(blending_tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            blending_vocab = data["model"]["vocab"]
            blending_tokens = list(blending_vocab.keys())
        if "added_tokens" in data:
            for token_info in data["added_tokens"]:
                token_id = token_info["id"]
                token_str = token_info["content"]
                blending_vocab[token_str] = token_id
        blending_tokens = list(blending_vocab.keys())
    elif os.path.exists(blending_vocab_json_path):
        with open(blending_vocab_json_path, "r", encoding="utf-8") as f:
            blending_vocab = json.load(f)
        blending_tokens = list(blending_vocab.keys())
    else:
        raise FileNotFoundError(
            f"Neither 'tokenizer.json' nor 'vocab.json' found in {args.base_model_name_or_path}"
        )
        
    print("base_vocab: ", len(base_vocab))
    print("blending_tokens: ", len(blending_vocab))
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[base_tokenizer.__class__]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_tokenizer.__class__
    ]
    print("base_model_special_token: ", base_model_special_token)
    print("blending_model_special_token: ", blending_model_special_token)
            
    blending_to_base_mapping = dict()

    with multiprocessing.Pool(64) as pool:
        mapping_args = [
            (x, base_tokens, blending_model_special_token, base_model_special_token)
            for x in blending_tokens
        ]
        results = list(
            tqdm.tqdm(
                pool.starmap(find_best_mapping, mapping_args),
                total=len(blending_tokens),))
    
    base_tokens_ids = base_vocab
    blending_tokens_ids = blending_vocab
    id_mapping = {}
    for tmp_x, best_mapping in results:
        # print(tmp_x, best_mapping)
        if best_mapping != "":
            blending_to_base_mapping[tmp_x] = best_mapping
            blending_id = blending_tokens_ids[tmp_x]
            base_id = base_tokens_ids[best_mapping]
            id_mapping[blending_id] = base_id
    
    print("blending_to_base_mapping: ",len(blending_to_base_mapping))
    print("id_mapping: ",len(id_mapping))
    cnt = 0
    for k, v in blending_to_base_mapping.items():
        if k == v:
            cnt += 1
    logger.info(
        f"Total tokens in blending vocab: {len(blending_tokenizer.get_vocab())},"
        f"Total tokens in blending to base mapping: {len(blending_to_base_mapping)},"
        f"Total best matched tokens: {cnt}."
    )
    
    os.makedirs(os.path.dirname(args.vocab_mapping_save_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.id_mapping_save_dir), exist_ok=True)
    
    with open(args.vocab_mapping_save_dir, "w") as fout:
        json.dump(blending_to_base_mapping, fout, indent=2)
        
    with open(args.id_mapping_save_dir, "w") as fout:
        json.dump(id_mapping, fout, indent=2)

    print("already dump to", args.id_mapping_save_dir)
