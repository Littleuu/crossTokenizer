import json
from pathlib import Path
from rouge_score import rouge_scorer
import string

# ---------- 配置 ----------
sinst_file = Path("/data/user/whx/CTKD/DSKD/data/sinst/6_10/valid.jsonl")
student_file = Path("/data/user/whx/CTKD/SEDI/outputs/dolly/qwen_gpt2/SFT_SEDI/bf16_r0.5_e7_b4x2_lr1e-4_K100/answers.jsonl")
baseline_file = Path("/data/user/whx/CTKD/SEDI/outputs/dolly/qwen_gpt2/SEDI/bf16_r0.5_e7_b4x2_lr1e-4_K100/answers.jsonl")
output_txt = Path("sinst_lower_than_baseline.txt")

# ---------- 文本预处理 ----------
def normalize_text(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = ' '.join(s.split())
    return s

# ---------- 初始化 rouge scorer ----------
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_max(pred, gts):
    pred_norm = normalize_text(pred)
    scores = []
    for gt in gts:
        if isinstance(gt, list):
            gt = "; ".join(gt)
        gt_norm = normalize_text(gt)
        score = scorer.score(gt_norm, pred_norm)['rougeL'].fmeasure
        scores.append(score)
    return max(scores)

# ---------- 读取数据 ----------
with open(sinst_file, "r", encoding="utf-8") as f:
    sinst_data = [json.loads(line) for line in f]
with open(student_file, "r", encoding="utf-8") as f:
    student_data = [json.loads(line) for line in f]
with open(baseline_file, "r", encoding="utf-8") as f:
    baseline_data = [json.loads(line) for line in f]

# ---------- 检查长度 ----------
assert len(sinst_data) == len(student_data) == len(baseline_data), "样本数量不一致！"

# ---------- 保存下降样本到文本 ----------
with open(output_txt, "w", encoding="utf-8") as f:
    for idx, (orig, stu, base) in enumerate(zip(sinst_data, student_data, baseline_data)):
        ground_truth = orig.get("output")
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]

        student_output = stu.get("text", "")
        baseline_output = base.get("text", "")

        student_score = rouge_max(student_output, ground_truth)
        baseline_score = rouge_max(baseline_output, ground_truth)

        if student_score < baseline_score:
            f.write(f"index: {idx}\n")
            f.write(f"instruction:\n{orig.get('instruction','')}\n")
            f.write(f"input:\n{orig.get('input','')}\n")
            f.write(f"ground_truth:\n{'; '.join(ground_truth)}\n")
            f.write(f"student_output:\n{student_output}\n")
            f.write(f"baseline_output:\n{baseline_output}\n")
            f.write(f"student_rougeL: {student_score:.4f}\n")
            f.write(f"baseline_rougeL: {baseline_score:.4f}\n")
            f.write(f"rougeL_diff: {baseline_score - student_score:.4f}\n")
            f.write("\n" + "-"*80 + "\n\n")  # 分割线
print(f"完成，总共保存 {output_txt}")