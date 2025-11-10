from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
import pandas as pd
import os
# import nltk
# nltk.download("punkt")

from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from utils import print_rank, save_rank, all_gather
import re
from fractions import Fraction
from rouge_metric import compute_metrics
import torch
import torch.distributed as dist
import deepspeed

import json

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model

from evaluate_dolly import prepare_dataset_main

torch.set_num_threads(4)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

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

def multioutput_f1_score(y_true, y_pred):
    if y_true.ndim == 2 and y_pred.ndim == 2:
        f1_scores = [f1_score(y_true[:, i], y_pred[:, i], average='macro') for i in range(y_true.shape[1])]
        return np.mean(f1_scores)
    else:
        return f1_score(y_true, y_pred, average='macro')

def run_model(args, tokenizer, model, dataset: PromptDataset, epoch, device):
    
    collate_fn = dataset.collate
    
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
    all_acc = []
    
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
        all_query_ids,
        all_response_ids,
        )

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def extract_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return float(num_numerator / num_denominator)
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return float(match.group().replace(',', ''))
        else:
            return None
    else:
        return None

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def process_results(completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        return False
    
def evaluate_math_and_gsm(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    query_ids, response_ids = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    all_preds = []
    all_responses = []
    result = []
    for q, r, a in zip(query_strs, response_strs, dataset.answers):
        if args.data_names in ["gsm8k"]:
            gt = a[0].replace("<n>", "\n").strip()
            temp_gt = gt.split('#### ')[1]
            temp_gt = int(temp_gt.replace(',', ''))
            
            preds = r.replace("<n>", "\n").strip()
            y_pred = extract_answer_number(preds)
            if y_pred != None:
                result.append(float(y_pred) == float(temp_gt))
            else:
                result.append(False)
                
            all_preds.append({
                    "ground truth": temp_gt,
                    "preds": preds,
                    "prompt":q.replace("<n>", "\n").strip(),
                })
            all_responses.append(r.replace("<n>", "\n").strip())
            
        if args.data_names in ["math"]:
            gt = a[0].replace("<n>", "\n").strip()
            temp_gt = remove_boxed(last_boxed_only_string(gt))
            preds = r.replace("<n>", "\n").strip()
            res = process_results(preds, temp_gt)
            result.append(res)
            all_preds.append({
                    "ground truth": temp_gt,
                    "preds": preds,
                    "prompt":q.replace("<n>", "\n").strip(),
                })
            all_responses.append(r.replace("<n>", "\n").strip())
            
        if args.data_names in ["metamath"]:
            gt = a[0].replace("<n>", "\n").strip()
            temp_gt = gt.split('The answer is: ')[1]
            preds = r.replace("<n>", "\n").strip()
            res = process_results(preds, temp_gt)
            result.append(res)
            all_preds.append({
                    "ground truth": temp_gt,
                    "preds": preds,
                    "prompt":q.replace("<n>", "\n").strip(),
                })
            all_responses.append(r.replace("<n>", "\n").strip())
        
        if args.data_names in ["orca"]:
            gt = a[0].replace("<n>", "\n").strip()
            temp_gt = gt.split('#### ')[1]
            temp_gt = float(temp_gt.replace(',', ''))
            preds = r.replace("<n>", "\n").strip()
            y_pred = extract_number(preds)
            if y_pred != None:
                result.append(float(y_pred) == float(temp_gt))
            else:
                result.append(False)
                
            all_preds.append({
                    "ground truth": temp_gt,
                    "preds": preds,
                    "prompt":q.replace("<n>", "\n").strip(),
                })
            all_responses.append(r.replace("<n>", "\n").strip())
            
    with open(os.path.join(args.save_dir, "answers.jsonl"), "w") as f:    
        json.dump(all_preds, f, indent=2)
    
    pass1_acc = sum(result) / len(result)
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    results = {
        "Pass@1": "{:.4f}".format(pass1_acc),
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
    if args.task == "eval_main":
        dataset = prepare_dataset_main(
            args,
            tokenizer,
        )
    else:
        raise NotImplementedError
    model = setup_model(args, ds_config, device)
    
    evaluate_math_and_gsm(args, tokenizer, model, dataset["test"], "test", 0, device)
    
if __name__ == "__main__":
    main()