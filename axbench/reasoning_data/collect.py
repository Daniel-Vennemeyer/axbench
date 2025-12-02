import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
OUTPUT_PATH = "axbench/reasoning_data/reasoning_data.jsonl"
USE_VLLM = False  # flip to True if using vLLM

# -----------------------------
# Load Model
# -----------------------------

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

if not USE_VLLM:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_NAME, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

# -----------------------------
# Reasoning Category Prompt
# -----------------------------

CLASSIFICATION_PROMPT = """You are an expert at classifying reasoning tasks.

Given the user question below, identify the SINGLE most appropriate reasoning domain category.
Use highly specific categories, such as:
- Number Theory Reasoning
- Basic Arithmetic Reasoning
- Organic Chemistry Reasoning
- Data Structures Reasoning
- Classical Mechanics Reasoning
- Linear Algebra Reasoning
- Computer Vision Reasoning
- Medieval European History Reasoning
- Renaissance Art History Reasoning
- Modern US Politics Reasoning
- Financial Accounting Reasoning
- Epidemiology Reasoning
- Constitutional Law Reasoning
- Cybersecurity Reasoning

Return ONLY the category name, nothing else.

User question:
\"\"\"{question}\"\"\""""

# -----------------------------
# Classification Function
# -----------------------------

def classify_reasoning_category(question: str) -> str:
    prompt = CLASSIFICATION_PROMPT.format(question=question)

    if USE_VLLM:
        outputs = llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("\n")[-1].strip()

# -----------------------------
# Helper: Extract messages
# -----------------------------

def extract_user_and_assistant(messages):
    """Extract user prompt and assistant response from the message list."""
    user_msg = None
    assistant_msg = None
    for m in messages:
        if m["role"] == "user" and user_msg is None:
            user_msg = m["content"].strip()
        if m["role"] == "assistant" and assistant_msg is None:
            assistant_msg = m["content"].strip()
    return user_msg, assistant_msg

# -----------------------------
# Load Old Dataset
# -----------------------------

ds = load_dataset("kenhktsui/longtalk-cot-v0.1")

# -----------------------------
# Convert + Categorize
# -----------------------------

concept_map = {}   # category → concept_id
next_concept_id = 0

output_rows = []

for ex in tqdm(ds["train"], desc="Classifying examples"):
    # Extract messages
    chosen_msg_list = ex["chosen"]
    user_input, assistant_output = extract_user_and_assistant(chosen_msg_list)

    # Run classification
    category = classify_reasoning_category(user_input)

    # Debug: print first 100 categories
    if next_concept_id < 100:
        print(f"Example {next_concept_id}: {category}")

    # Assign stable concept ID
    if category not in concept_map:
        concept_map[category] = next_concept_id
        next_concept_id += 1

    concept_id = concept_map[category]

    # Build final record
    record = {
        "input": user_input,
        "output": assistant_output,
        "output_concept": category,
        "concept_genre": "positive",
        "dataset_category": "instruction",
        "concept_id": concept_id
    }

    output_rows.append(record)

# -----------------------------
# Save Final Output
# -----------------------------

with open(OUTPUT_PATH, "w") as f:
    for ex in output_rows:
        f.write(json.dumps(ex) + "\n")

print("Done! Wrote", OUTPUT_PATH)
print("Discovered categories:")
for k, v in concept_map.items():
    print(f"{v}: {k}")