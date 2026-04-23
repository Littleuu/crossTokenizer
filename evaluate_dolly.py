from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig

import pandas as pd
import os
import torch
import torch.distributed as dist
import deepspeed

import json

from arguments import get_args

from sklearn.metrics import f1_score
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# import nltk
# nltk.download("punkt")
from utils import get_tokenizer, get_model, set_random_seed
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from utils import initialize, print_args, log_rank
import torch
import torch.distributed as dist
import json
from utils import print_rank, save_rank, all_gather

from rouge_metric import compute_metrics

torch.set_num_threads(4)


def prepare_dataset_main(args, tokenizer):
    data = {}
    data["test"] = PromptDataset(
        args, 
        tokenizer, 
        "valid", 
        data_path=args.data_dir, 
        num=args.dev_num
    )

    return data


def run_model(args, tokenizer, model, dataset: PromptDataset, epoch, device):
    
    collate_fn = dataset.collate
    
    # if args.model_parallel:
    #     dp_world_size = mpu.get_data_parallel_world_size()
    #     dp_rank = mpu.get_data_parallel_rank()
    #     dp_group = mpu.get_data_parallel_group()
    # else:
    dp_world_size = dist.get_world_size()
    dp_rank = dist.get_rank()
    dp_group = None
    
    sampler = DistributedSampler(
        dataset, 
        shuffle=False, 
        drop_last=False, 
        rank=dp_rank, 
        num_replicas=dp_world_size
    )
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn
    )
    model.eval()
    
    all_query_ids = []
    all_response_ids = []
    all_lm_losses = []
    
    generation_config = GenerationConfig(
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {args.data_names} ", disable=(dist.get_rank() != 0))):
            if it == 0:
                print_rank("############### Example ###############")
                print_rank(tokenizer.decode(model_batch["input_ids"][0], skip_special_tokens=True))
                print_rank("############### End ###############")
            
            dataset.move_to_device(model_batch, no_model_batch, device)

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids == tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
                # print(input_ids[0], position_ids[0], attention_mask[0])
                out = model(
                    input_ids=input_ids, 
                    position_ids=position_ids, 
                    attention_mask=attention_mask, 
                    return_dict=True
                )
            else:
                out = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    return_dict=True
                )
            logits = out.logits
            loss_mask = (label_ids != -100).float()
            loss_func = nn.CrossEntropyLoss(reduction="none")
            lm_loss = loss_func(
                logits.view(-1, logits.size(-1)), 
                label_ids.view(-1)
            ).view(label_ids.size())
            lm_loss_mean = torch.sum(lm_loss * loss_mask, -1) / torch.sum(loss_mask, -1)
            all_lm_losses.append(lm_loss_mean)
 
            query_ids = model_batch["input_ids"]
            max_new_tokens = args.max_length - query_ids.size(1)

            gen_out = model.generate(
                **model_batch,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
            full_ids = gen_out.sequences
            response_ids = full_ids[:, query_ids.size(1):] # remove prompt (may include start token)
            
            query_ids = F.pad(
                query_ids, 
                (args.max_prompt_length-query_ids.size(1), 0, 0, 0), 
                value=tokenizer.pad_token_id
            )
            response_ids = F.pad(
                response_ids, 
                (0, args.max_length-args.max_prompt_length-response_ids.size(1), 0, 0), 
                value=tokenizer.pad_token_id
            )
            
            all_query_ids.append(query_ids)
            all_response_ids.append(response_ids)

    all_lm_losses = torch.cat(all_lm_losses)
    mean_lm_loss = all_lm_losses.mean()
    dist.all_reduce(mean_lm_loss, dist.ReduceOp.SUM, group=dp_group)
    mean_lm_loss = mean_lm_loss.item() / dp_world_size
        
    all_query_ids = torch.cat(all_query_ids)
    all_query_ids = all_gather(
        all_query_ids, 
        dim=1, 
        group=dp_group, 
        world_size=dp_world_size, 
        op="stack"
    )
    all_query_ids = all_query_ids.view(-1, all_query_ids.size(-1))
    all_query_ids = all_query_ids[:len(dataset)]
    
    all_response_ids = torch.cat(all_response_ids)
    all_response_ids = all_gather(
        all_response_ids, 
        dim=1, 
        group=dp_group, 
        world_size=dp_world_size, 
        op="stack"
    )
    all_response_ids = all_response_ids.view(-1, all_response_ids.size(-1))
    all_response_ids = all_response_ids[:len(dataset)]
        
    return (
        mean_lm_loss,
        all_query_ids,
        all_response_ids
        )

def evaluate_main(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    lm_loss, query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    with open(os.path.join(args.save_dir, "preds.txt"), "w", encoding="utf-8") as f:
        for q, r in zip(query_strs, response_strs):
            f.write(q.replace("\n", "<n>") + "\t\t" + r.replace("\n", "<n>") + "\n")

    all_preds = [[]]
    for q, r in zip(query_strs, response_strs):
        all_preds[0].append((q, q + r))
    torch.save(all_preds, os.path.join(args.save_dir, "preds.pt"))

    all_responses = []
    with open(os.path.join(args.save_dir, "answers.jsonl"), "w") as f:    
        for p in all_preds[0]:
            q, r = p
            r = r[len(q):]
            idx = r.find("<|endoftext|>")
            if idx >= 0:
                r = r[:idx]
            f.write(json.dumps({
                "text": r.replace("<n>", "\n").strip()
            }) + "\n")
            all_responses.append(r.replace("<n>", "\n").strip())
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    results = {
        "rougeL": "{:.4f}".format(gen_res["rougeL"]),
        "exact_match": gen_res["exact_match"],
        "f1": "{:.4f}".format(gen_res["f1"]),
        "bleu": "{:.4f}".format(gen_res["bleu"]),
        "fluency": "{:.4f}".format(gen_res["fluency"]),
        "lm_loss": "{:.4f}".format(lm_loss),
        "avg_gen_length": mean_gen_length,
        "params": str(args.save_dir).split("/")[-2],
        "seed": args.seed,
        "dataset": args.data_names,
        "method": str(args.save_dir).split("/")[-3],
    }

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    log_str = f"{split} | name: {args.data_names} | {results} | lm_loss {round(lm_loss, 4)} | avg. gen lenth: {mean_gen_length} | seed {args.seed}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save_dir, "log.txt"))

def maybe_init_distributed(args):
    """Safe 初始化分布式（支持 torchrun / deepspeed）。"""
    if dist.is_available() and not dist.is_initialized():
        if args.deepspeed:
            deepspeed.init_distributed()
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
    # 设定本地 GPU
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def get_tokenizer(args):
    """稳健加载 tokenizer，并保证 pad/eos 一定可用。"""
    try:
        tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        print(f"[INFO] Loaded tokenizer via AutoTokenizer.from_pretrained({args.model_path})")
    except Exception:
        tok_file = os.path.join(args.model_path, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(f"tokenizer.json not found in {args.model_path}")
        print(f"[INFO] Loading tokenizer directly from {tok_file}")
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)

    # 不同模型类型的 pad/eos 兜底
    if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama"]:
        # GPT/LLaMA 默认无 pad，用 eos 代替
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "<|endoftext|>"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        if tok.eos_token_id is None:
            # GPT-2 默认
            tok.eos_token_id = 50256 if tok.convert_tokens_to_ids(tok.eos_token) is None else tok.convert_tokens_to_ids(tok.eos_token)
    elif args.model_type == "qwen":
        # Qwen：若缺失则兜底为官方 id
        if tok.eos_token_id is None:
            tok.eos_token_id = 151643
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

    # 最终兜底，确保不为 None
    if tok.eos_token_id is None:
        tok.eos_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    print(f"[INFO] eos_token_id = {tok.eos_token_id}, pad_token_id = {tok.pad_token_id}")
    return tok


def pick_available_split(data_dir: str, model_type: str, use_json: bool) -> str:
    """自动选择可用 split：优先 test，其次 valid/dev/train。支持 _{model_type}.jsonl 变体。"""
    if use_json:
        candidates = [
            (f"test_{model_type}.jsonl", "test"),
            ("test.jsonl", "test"),
            ("valid.jsonl", "valid"),
            ("dev.jsonl", "dev"),
            ("train.jsonl", "train"),
        ]
    else:
        candidates = [
            ("test.txt", "test"),
            ("valid.txt", "valid"),
            ("dev.txt", "dev"),
            ("train.txt", "train"),
        ]
    for fname, split in candidates:
        if os.path.exists(os.path.join(data_dir, fname)):
            return split
    raise FileNotFoundError(
        f"No dataset file found in {data_dir}. Expected one of: "
        f"{', '.join([c[0] for c in candidates])}"
    )


def load_ds_config(path: str):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # 1) 参数
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 2) 分布式初始化（先 init，再 set_random_seed，避免 get_rank 报错）
    local_rank = maybe_init_distributed(args)

    # 3) 随机种子
    set_random_seed(args.seed)

    # 4) 设备
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # 5) tokenizer
    tokenizer = get_tokenizer(args)

    # 6) DeepSpeed 配置（可选）
    ds_config = load_ds_config(args.deepspeed_config) if args.deepspeed else None

    # 7) 模型
    model = get_model(args, device)
    if args.deepspeed:
        model, _, _, _ = deepspeed.initialize(model=model, config=ds_config, model_parameters=model.parameters())

    # 8) 数据 split 自动回退（你没有 test.jsonl 时会自动用 valid/dev/train）
    split = pick_available_split(args.data_dir, args.model_type, args.json_data)
    log_rank(f"Using split: {split}")

    dataset = PromptDataset(
        args=args,
        tokenizer=tokenizer,
        split=split,              # 这里用我们挑出的可用 split
        data_path=args.data_dir,
        num=args.dev_num,
    )

    # 9) 评估
    if args.do_eval:
        # 第 2 个参数只是标签名，仅用于日志/保存；为了直观，用实际 split 名
        evaluate_main(args, tokenizer, model, dataset, split, 0, device)


if __name__ == "__main__":
    main()

