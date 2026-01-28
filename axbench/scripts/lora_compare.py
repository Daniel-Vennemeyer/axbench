# ==============================
# Minimal LoRA benchmarking script
# ==============================

import argparse
import json
import math
import re
from pathlib import Path

import torch
from peft import PeftModel
from huggingface_hub import hf_hub_download
from peft import PeftConfig
import json as _json
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import inspect
from peft.tuners.lora import LoraConfig

# -----------------------------
# Parsing helpers
# -----------------------------

def parse_gsm8k_gold(answer):
    if "####" in answer:
        answer = answer.split("####")[-1]
    nums = re.findall(r"-?\d+", answer.replace(",", ""))
    return int(nums[-1]) if nums else None

def parse_gsm8k_pred(text):
    """
    Parse the predicted GSM8K answer robustly.

    Priority order:
      1) If the model outputs a GSM8K-style marker, prefer the integer after '####'.
      2) If the text contains an '=', take the substring after the last '=' and extract the first integer (allow $/paren/comma/punct).
      3) Scan lines bottom-up and return the first line that is a *pure integer*.
      4) Fallback: return the last "answer-like" integer near the end of the text,
         avoiding common pitfalls:
           - decimals (e.g. 0.67 should not yield 67)
           - digits glued to letters (e.g. mwm654)
           - intermediate fractions (e.g. 3/60)
         while allowing trailing punctuation like '$460.'.
    """
    if text is None:
        return None

    clean = text.replace(",", "")
    # 1) GSM8K marker
    if "####" in clean:
        tail = clean.split("####")[-1]
        m = re.search(r"-?\d+", tail)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass

    # 2) Highest-priority: after last '=' extract first integer (allow $/paren/commas/trailing punct)
    if "=" in clean:
        tail = clean.split("=")[-1]
        # Remove leading/trailing whitespace and possible currency/paren
        tail = tail.strip()
        # Look for $ or ( or whitespace, then integer, possibly with trailing . or )
        # e.g. = $460. or = (460)
        m = re.search(r"[\$\(\s]*(-?\d+)", tail)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    # 3) Integer-only line (canonical)
    lines = clean.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)

    # 4) Fallback: scan integer tokens with context-aware filtering
    # Token pattern: optional sign + digits, bounded so it's not glued to letters/digits.
    # Allow trailing punctuation (.,;:!?) and currency symbols around it.
    token_iter = list(re.finditer(r"(?<![A-Za-z0-9])(-?\d+)(?![A-Za-z0-9])", clean))
    if not token_iter:
        return None

def parse_supergpqa_pred(text):
    if text is None:
        return None
    m = re.search(r"\b([A-J])\b", text.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None

# -----------------------------
# Stats
# -----------------------------

def binomial_ci_95(correct, total):
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

# -----------------------------
# Prompts
# -----------------------------

GSM8K_PROMPT = (
    "Solve the following math problem step by step.\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

SUPERGPQA_PROMPT = (
    "Answer the following multiple choice question.\n\n"
    "Question: {question}\n\n"
    "{options}\n\n"
    "Answer:"
)

# -----------------------------
# Benchmark runners
# -----------------------------

@torch.no_grad()
def run_gsm8k(model, tokenizer, max_questions, batch_size, device):
    dataset = load_dataset("gsm8k", "main", split="test")
    if max_questions:
        dataset = dataset.select(range(max_questions))

    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="GSM8K"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [
            GSM8K_PROMPT.format(question=ex["question"])
            for ex in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for ex, out in zip(batch, decoded):
            gold = parse_gsm8k_gold(ex["answer"])
            pred = parse_gsm8k_pred(out)
            if gold is not None and pred == gold:
                correct += 1
            total += 1

    acc = correct / total
    ci = binomial_ci_95(correct, total)
    return acc, ci, correct, total

@torch.no_grad()
def run_supergpqa(model, tokenizer, max_questions, batch_size, device, discipline=None, field=None):
    """
    SuperGPQA loader that bypasses HuggingFace Datasets entirely.
    Reads JSONL directly from the Hub to avoid Python 3.12 + datasets crashes.
    """

    # Download raw JSONL
    jsonl_path = hf_hub_download(
        repo_id="m-a-p/SuperGPQA",
        filename="train.jsonl",
        repo_type="dataset",
    )

    correct = 0
    total = 0
    batch = []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="SuperGPQA"):
            ex = json.loads(line)

            # Discipline / field filtering
            if discipline is not None:
                if str(ex.get("discipline", "")).strip().lower() != discipline.strip().lower():
                    continue

            if field is not None:
                if str(ex.get("field", "")).strip().lower() != field.strip().lower():
                    continue

            batch.append(ex)
            if max_questions is not None and total + len(batch) >= max_questions:
                batch = batch[: max_questions - total]

            if len(batch) < batch_size:
                if max_questions is not None and total + len(batch) >= max_questions:
                    pass
                else:
                    continue

            # Process batch
            prompts = []
            for ex in batch:
                options = "\n".join(
                    f"{chr(ord('A') + j)}. {opt}"
                    for j, opt in enumerate(ex["options"])
                )
                prompts.append(
                    SUPERGPQA_PROMPT.format(
                        question=ex["question"],
                        options=options
                    )
                )

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for ex, out in zip(batch, decoded):
                pred = parse_supergpqa_pred(out)
                if pred == ex["answer_letter"]:
                    correct += 1
                total += 1

            batch.clear()

            if max_questions is not None and total >= max_questions:
                break

    acc = correct / total if total > 0 else 0.0
    ci = binomial_ci_95(correct, total)
    return acc, ci, correct, total

# -----------------------------
# LoRA evaluation plan
# -----------------------------

LORA_EVAL_PLAN = [
    # GSM8K
    {
        "lora": "marco-molinari/axbench-lora-basic_arithmetic_reasoning",
        "benchmark": "gsm8k",
    },

    # SuperGPQA
    {
        "lora": "marco-molinari/axbench-lora-epidemiology_reasoning",
        "benchmark": "supergpqa",
        "discipline": "Medicine",
    },
    {
        "lora": "marco-molinari/axbench-lora-classical_mechanics_reasoning",
        "benchmark": "supergpqa",
        "field": "Physics",
    },
    {
        "lora": "marco-molinari/axbench-lora-organic_chemistry_reasoning",
        "benchmark": "supergpqa",
        "field": "Chemistry",
    },
    {
        "lora": "marco-molinari/axbench-lora-medieval_european_history_reasoning",
        "benchmark": "supergpqa",
        "discipline": "History",
    },
    {
        "lora": "marco-molinari/axbench-lora-constitutional_law_reasoning",
        "benchmark": "supergpqa",
        "discipline": "Legal",
    },
    {
        "lora": "marco-molinari/axbench-lora-narrative_structure_reasoning",
        "benchmark": "supergpqa",
        "discipline": "Literature and Arts",
    },
]

# -----------------------------
# Helper to robustly load LoRA
# -----------------------------

def load_peft_model_robust(base_model, lora_repo):
    """
    Load a PEFT / LoRA adapter, stripping unknown config fields
    (e.g. `corda_config`) for older peft versions.
    """
    try:
        return PeftModel.from_pretrained(base_model, lora_repo)
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise

        # Download and sanitize adapter_config.json
        cfg_path = hf_hub_download(lora_repo, "adapter_config.json")
        with open(cfg_path, "r") as f:
            cfg = _json.load(f)

        # Drop any keys not accepted by this PEFT version's LoraConfig
        valid_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        valid_keys.discard("self")

        cfg = {k: v for k, v in cfg.items() if k in valid_keys}

        # Write sanitized config to a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            clean_cfg_path = Path(tmpdir) / "adapter_config.json"
            with open(clean_cfg_path, "w") as f:
                _json.dump(cfg, f)

            # The sanitized adapter_config.json now only contains keys
            # supported by the local PEFT version.

            # Load using explicit config
            peft_cfg = PeftConfig.from_pretrained(tmpdir)
            return PeftModel.from_pretrained(
                base_model,
                lora_repo,
                config=peft_cfg,
            )

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/gemma-2-2b-it")
    parser.add_argument("--lora_models", nargs="+", required=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--out_dir", default="lora_benchmarks")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.use_bf16 else None,
        device_map="auto"
    ).eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in LORA_EVAL_PLAN:
        lora_path = spec["lora"]
        benchmark = spec["benchmark"]
        discipline = spec.get("discipline")
        field = spec.get("field")

        print(f"\n=== Evaluating LoRA: {lora_path} ===")

        model = load_peft_model_robust(base_model, lora_path)
        model = model.eval()

        if benchmark == "gsm8k":
            acc, ci, correct, total = run_gsm8k(
                model,
                tokenizer,
                args.max_questions,
                args.batch_size,
                device,
            )
        else:
            acc, ci, correct, total = run_supergpqa(
                model,
                tokenizer,
                args.max_questions,
                args.batch_size,
                device,
                discipline=discipline,
                field=field,
            )

        summary = {
            "lora": lora_path,
            "benchmark": benchmark,
            "discipline": discipline,
            "field": field,
            "accuracy": acc,
            "ci_95": {"low": ci[0], "high": ci[1]},
            "correct": correct,
            "total": total,
        }

        with open(out_dir / f"{benchmark}_{Path(lora_path).name}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(summary)


if __name__ == "__main__":
    main()
