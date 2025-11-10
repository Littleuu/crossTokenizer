from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
import pandas as pd
import os
# import nltk
# nltk.download("punkt")
import torch.distributed as dist
import deepspeed
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, all_gather
import re
from fractions import Fraction
from rouge_metric import compute_metrics
torch.set_num_threads(4)
from typing import List, Union, Iterable, Dict
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import gzip
import tempfile
import subprocess
from execution import check_correctness
from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
}

def read_problems(evalset_file: str = "humanevalpack.jsonl.gz") -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def multioutput_f1_score(y_true, y_pred):
    if y_true.ndim == 2 and y_pred.ndim == 2:
        f1_scores = [f1_score(y_true[:, i], y_pred[:, i], average='macro') for i in range(y_true.shape[1])]
        return np.mean(f1_scores)
    else:
        return f1_score(y_true, y_pred, average='macro')

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def batchify(examples, tokenizer, max_length, device):
    # left pad
    inputs = tokenizer(examples, max_length=max_length, truncation=True, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids']
    attn_mask = input_ids.ne(tokenizer.pad_token_id).float()
    return {"input_ids": input_ids.to(device), "attention_mask": attn_mask.to(device)}

def read_dataset(
    data_file: str = None,
    dataset_type: str = "humaneval",
    num_shot=None,
) -> Dict:
    """
    Reads a dataset and returns a dictionary of tasks.
    """
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset

def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = "humanevalpack.jsonl.gz",
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    tmp_dir = None
    problems = read_dataset(problem_file,
                            dataset_type="humaneval") 

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            sample["test_code"] = sample["completion"]
            args = (task_id, sample, "python", timeout, tmp_dir, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    return pass_at_k

def extract_code_blocks(text):
    # code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    # return [code.strip() for code in code_blocks]
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else ""
          
def evaluate_code(args, tokenizer, model, device):
    valid_path = os.path.join(args.data_dir, "valid.jsonl")
    if os.path.exists(valid_path):
        with open(valid_path) as f:
            raw_data = [json.loads(l) for l in f.readlines()]
        if args.dev_num != -1:
            raw_data = raw_data[:args.dev_num]
        examples = [" " + x["prompt"].rstrip() if "t5" in args.model_path else x["prompt"].rstrip() for x in raw_data]
        references = [x["output"] for x in raw_data]
        task_ids = [x["task_id"] for x in raw_data]
    else:
        raise FileNotFoundError(f"No such file named {args.data_path}")
    src_chunks = list(chunks(examples, args.eval_batch_size))
    ref_chunks = list(chunks(references, args.eval_batch_size))
    tasks_chunks = list(chunks(task_ids, args.eval_batch_size))
    
    tokenizer.padding_side = "left"
    all_responses = []
    all_completions = []
    for examples_chunk, examples_ref_chunk, task_ids in tqdm.tqdm(zip(src_chunks, ref_chunks, tasks_chunks), total=len(src_chunks)):
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            examples_chunk = [text.rstrip('<|endoftext|>') for text in examples_chunk]
        with torch.no_grad():                  
            batch = batchify(examples_chunk, tokenizer, args.max_prompt_length, device)
            slen = batch["input_ids"].size(1)
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

            query_ids = batch["input_ids"]
            max_new_tokens = args.max_length - slen
            
            gen_out = model.generate(
                **batch,
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
            
            query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
            response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            for q, r, gt, task_id in zip(query_strs, response_strs, examples_ref_chunk, task_ids):
                completion = extract_code_blocks(r)
                all_completions.append({
                    "task_id": task_id,
                    "ground truth": gt,
                    "preds": r,
                    "prompt":q.replace("<n>", "\n").strip(),
                    "completion": completion,
                })
                all_responses.append(r)
    
    gen_res = compute_metrics(all_responses, references)
    
    save_file = os.path.join(args.save_dir, "answers.jsonl")
    with open(save_file, "w") as f:    
        for item in all_completions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    pass1_acc = evaluate_functional_correctness(save_file, k=[1], n_workers=4, timeout=3.0, problem_file=os.path.join(args.data_dir, "humanevalpack.jsonl.gz")) 
    print(pass1_acc)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in all_responses])

    results = {
        "pass@1": "{:.4f}".format(pass1_acc["pass@1"]),
        "rougeL": "{:.4f}".format(gen_res["rougeL"]),
        "exact_match": gen_res["exact_match"],
        "f1": "{:.4f}".format(gen_res["f1"]),
        "bleu": "{:.4f}".format(gen_res["bleu"]),
        "fluency": "{:.4f}".format(gen_res["fluency"]),
        "avg_gen_length": mean_gen_length,
        "params": str(args.save_dir).split("/")[-2],
        "seed": args.seed,
        "dataset": args.data_names,
        "method": str(args.save_dir).split("/")[-3],
    }
    print(results)
    df = pd.DataFrame(results, index=[0])
    df.to_excel(os.path.join(args.save_dir, "eval.xlsx"), index=False)


def setup_model(args, ds_config, device):
    model = get_model(args, device)
    optimizer, lr_scheduler = None, None
        
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config
    )
    
    # get the memory usage
    return model


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)
    
    if dist.get_rank() == 0:
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0

    args.fp32 = not ds_config["fp16"]["enabled"] 
    args.deepspeed_config = None

    # get the tokenizer
    tokenizer = get_tokenizer(args)
    model = setup_model(args, ds_config, device)
    
    evaluate_code(args, tokenizer, model, device)
    
if __name__ == "__main__":
    main()