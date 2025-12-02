BATCH_SIZE = 4
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

import re

def clean_category(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    if "\n" in text:
        text = text.split("\n")[-1].strip()
    text = re.sub(r'(?i).*?category is:?', '', text).strip()
    return text

def is_valid_category(text: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9 ]+", text):
        return False
    if not text.endswith("Reasoning"):
        return False
    if len(text.split()) < 2:
        return False
    return True


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
    messages = [
        {"role": "system", "content": "You are an expert at classifying reasoning tasks."},
        {"role": "user", "content": CLASSIFICATION_PROMPT.format(question=question)}
    ]

    if USE_VLLM:
        outputs = llm.generate(CLASSIFICATION_PROMPT.format(question=question), sampling_params)
        return outputs[0].outputs[0].text.strip()

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=32,
            do_sample=False,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only assistant content
    if "assistant" in decoded:
        decoded = decoded.split("assistant")[-1]

    # Keep only last non-empty line
    category = clean_category(decoded)

    tries = 0
    while not is_valid_category(category) and tries < 5:
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=32,
                do_sample=False,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in decoded:
            decoded = decoded.split("assistant")[-1]
        category = clean_category(decoded)
        tries += 1

    return category


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

batch_inputs = []
batch_assistants = []

for ex in tqdm(ds["train"], desc="Classifying examples"):
    user_input, assistant_output = extract_user_and_assistant(ex["chosen"])
    batch_inputs.append(user_input)
    batch_assistants.append(assistant_output)

    if len(batch_inputs) == BATCH_SIZE:
        # classify as batch
        categories = [classify_reasoning_category(q) for q in batch_inputs]

        for i in range(BATCH_SIZE):
            category = categories[i]

            if next_concept_id < 100:
                print(f"Example {next_concept_id}: {category}")

            if category not in concept_map:
                concept_map[category] = next_concept_id
                next_concept_id += 1

            record = {
                "input": batch_inputs[i],
                "output": batch_assistants[i],
                "output_concept": category,
                "concept_genre": "positive",
                "dataset_category": "instruction",
                "concept_id": concept_map[category]
            }
            output_rows.append(record)

        batch_inputs = []
        batch_assistants = []

# process remainder
for i in range(len(batch_inputs)):
    category = classify_reasoning_category(batch_inputs[i])

    if next_concept_id < 100:
        print(f"Example {next_concept_id}: {category}")

    if category not in concept_map:
        concept_map[category] = next_concept_id
        next_concept_id += 1

    record = {
        "input": batch_inputs[i],
        "output": batch_assistants[i],
        "output_concept": category,
        "concept_genre": "positive",
        "dataset_category": "instruction",
        "concept_id": concept_map[category]
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