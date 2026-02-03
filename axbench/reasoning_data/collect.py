import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 4  # increase for better GPU utilization (adjust to 32 if memory allows)
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------
# Paths
# -----------------------------
OUTPUT_PATH = "axbench/reasoning_data/reasoning_data.jsonl"

# -----------------------------
# Load Model
# -----------------------------

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
    )

model.eval()

import re

def normalize_category(cat: str) -> str:
    cat = cat.strip().lower()
    cat = re.sub(r"\s+", " ", cat)
    cat = cat.replace("&", "and")
    return cat

def clean_category(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = text.split("\n")[0].strip()
    text = re.sub(r"^[^A-Za-z]*", "", text)
    text = re.sub(r"[^A-Za-z &\-]", "", text)
    return text


# -----------------------------
# Reasoning Category Prompt
# -----------------------------

CLASSIFICATION_PROMPT = """You are an expert academic indexer.

Your task is to assign the question below to the SINGLE most relevant academic or professional FIELD.

Instructions:
- Classify by subject matter only.
- Do NOT describe the type of reasoning.
- Do NOT include words like “Reasoning”, “Analysis”, or “Thinking”.
- Choose the field that a university department or textbook would place this question in.

Process (do this silently):
1) Identify what the question is fundamentally ABOUT.
2) Identify the discipline that studies that subject.

Example Medical-field classifications:
- Use “Clinical Medicine” for diagnosis, treatment, symptoms, physiology, or patient care.
- Use “Pharmacology” for drugs, mechanisms, dosing, or interactions.
- Use “Epidemiology” if the question is about populations, prevalence, incidence, risk factors, or disease spread.
- More examples: use “Genetics”, “Neuroscience”, “Public Health”, “Pathology”, “Immunology”, etc.

If no standard field fits exactly, invent a reasonable and concise field name.

The field name should be:
- 1–4 words
- A noun phrase
- A standard academic discipline or subfield

Question:
''{question}''

Return ONLY the field name:"""

# -----------------------------
# Classification Function
# -----------------------------

def classify_reasoning_batch(questions):
    messages = [
        [
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(question=q)}
        ]
        for q in questions
    ]

    enc = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        return_dict=True
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(
        outputs[:, input_ids.shape[1]:],
        skip_special_tokens=True
    )

    categories = []
    for text in decoded:
        category = clean_category(text)
        category = normalize_category(category)
        categories.append(category)

    return categories


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
batch_negatives = []

for ex in tqdm(ds["train"], desc="Classifying examples"):
    user_input, assistant_output = extract_user_and_assistant(ex["chosen"])
    _, negative_output = extract_user_and_assistant(ex["rejected"])
    batch_inputs.append(user_input)
    batch_assistants.append(assistant_output)
    batch_negatives.append(negative_output)

    if len(batch_inputs) == BATCH_SIZE:
        # classify as batch
        categories = classify_reasoning_batch(batch_inputs)

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
                "output_negative": batch_negatives[i],
                "output_concept": category,
                "concept_genre": "positive",
                "dataset_category": "instruction",
                "concept_id": concept_map[category]
            }
            output_rows.append(record)

        batch_inputs = []
        batch_assistants = []
        batch_negatives = []

# process remainder as a single batch
if batch_inputs:
    categories = classify_reasoning_batch(batch_inputs)
    for i, category in enumerate(categories):
        if next_concept_id < 100:
            print(f"Example {next_concept_id}: {category}")

        if category not in concept_map:
            concept_map[category] = next_concept_id
            next_concept_id += 1

        record = {
            "input": batch_inputs[i],
            "output": batch_assistants[i],
            "output_negative": batch_negatives[i],
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