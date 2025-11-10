from data_utils.prompt_datasets import PromptDataset
from transformers import GenerationConfig
import pandas as pd
import os
import torch
import torch.distributed as dist
import deepspeed

import json

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model

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

            all_ids = torch.cat([model_batch["input_ids"], no_model_batch["rest_ids"]], dim=-1)
            input_ids = all_ids[:, :-1]
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            label_ids = all_ids[:, 1:]
            label_ids = torch.masked_fill(label_ids, label_ids == tokenizer.pad_token_id, -100)
            label_ids[:, :model_batch["input_ids"].size(1)-1] = -100
            if args.model_type in ["gpt2"]:
                position_ids = (torch.cumsum(attention_mask, dim=-1) - 1) * attention_mask
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
            
            pad_mask = label_ids.ne(-100)
            token_acc_num = logits.argmax(-1).eq(label_ids).float()
            token_acc_num = token_acc_num.masked_fill_(~pad_mask, 0.0).sum()/ pad_mask.sum()
            probs = logits.softmax(-1)
            top1_prob = probs.max(-1)[0].masked_fill(~pad_mask, 0.0).sum()
            all_acc.append(token_acc_num)
 
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
    
    mean_acc = sum(all_acc) / len(all_acc) * 100.0
        
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
        all_response_ids,
        mean_acc
        )


def evaluate_main(args, tokenizer, model, dataset: PromptDataset, split, epoch, device):
    lm_loss, query_ids, response_ids, mean_acc = run_model(args, tokenizer, model, dataset, epoch, device)
    query_strs = tokenizer.batch_decode(query_ids, skip_special_tokens=True)
    response_strs = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    
    all_preds = []
    all_responses = []
    for q, r, a in zip(query_strs, response_strs, dataset.answers):
        all_preds.append({
                "ground truth": a[0].replace("<n>", "\n").strip(),
                "preds": r.replace("<n>", "\n").strip(),
                "prompt":q.replace("<n>", "\n").strip(),
            })
        all_responses.append(r.replace("<n>", "\n").strip())

    with open(os.path.join(args.save_dir, "answers.jsonl"), "w") as f:    
        json.dump(all_preds, f, indent=2)
    
    gen_res = compute_metrics(all_responses, dataset.answers)

    mean_gen_length = np.mean([len(tokenizer.encode(s)) for s in response_strs])

    results = {
        "rougeL": "{:.4f}".format(gen_res["rougeL"]),
        "exact_match": gen_res["exact_match"],
        "f1": "{:.4f}".format(gen_res["f1"]),
        "bleu": "{:.4f}".format(gen_res["bleu"]),
        "fluency": "{:.4f}".format(gen_res["fluency"]),
        "acc": "{:.4f}".format(mean_acc),
        "lm_loss": "{:.4f}".format(lm_loss),
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
    
    evaluate_main(args, tokenizer, model, dataset["test"], "test", 0, device)
    
    
if __name__ == "__main__":
    main()