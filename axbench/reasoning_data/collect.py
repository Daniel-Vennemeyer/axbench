import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 4  # increase for better GPU utilization (adjust to 32 if memory allows)
FIELD_BATCH_SIZE = 16  # tune: 8–32 depending on GPU memory
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

import argparse
import os
import torch.multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
OUTPUT_PATH = f"axbench/reasoning_data/reasoning_data_shard{args.shard_id}.jsonl"

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

from transformers import LogitsProcessor

class ClosedSetLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed)] = 0.0
        return scores + mask


# -----------------------------
# Closed-set fields (expand as needed)
# -----------------------------

ALLOWED_FIELDS = [
    "mathematics",
    "computer science",
    "statistics",
    "probability theory",
    "linear algebra",
    "calculus",
    "number theory",
    "algebra",
    "geometry",
    "machine learning",
    "data science",
    "artificial intelligence",
    "programming languages",
    "software engineering",
    "electrical engineering",
    "mechanical engineering",
    "robotics",
    "control systems",
    "physics",
    "chemistry",
    "biology",
    "genetics",
    "neuroscience",
    "clinical medicine",
    "epidemiology",
    "public health",
    "psychology",
    "cognitive science",
    "economics",
    "finance",
    "accounting",
    "political science",
    "sociology",
    "linguistics",
    "philosophy",
    "history",
    "literature",
    "creative writing",
    "composition studies",
    "education",
    "law",
    "business",
    "marketing",
    "operations research",
    "game design",
    "architecture",
    "urban planning",
    "geography",
    "environmental science",
    "climatology",
    "ethics",
    "logic",
    "anthropology",
    "archeology",
    "astronomy",
    "astrophysics",
    "geology",
    "oceanography",
    "meteorology",
    "materials science",
    "biochemistry",
    "immunology",
    "pharmacology",
    "pathology",
    "nutrition",
    "forensic science",
    "communications",
    "media studies",
    "journalism",
    "film studies",
    "theater studies",
    "music theory",
    "art history",
    "graphic design",
    "fashion design",
    "culinary arts",
    "sports science",
    "kinesiology",
    "physical therapy",
    "occupational therapy",
    "nursing",
    "dentistry",
    "veterinary medicine",
    "social work",
    "international relations",
    "development studies",
    "religious studies",
    "theology",
    "archival studies",
    "museum studies",
    "library science",
    "supply chain management",
    "human resources",
    "project management",
    "quality assurance",
    "risk management",
    "cybersecurity",
    "network engineering",
    "cloud computing",
    "blockchain technology",
    "quantum computing",
    "virtual reality",
    "augmented reality",
    "game theory",
    "operations management",
    "strategic management",
    "entrepreneurship",
    "innovation management",
    "organizational behavior",
    "leadership studies",
    "conflict resolution",
    "negotiation studies",
    "customer relationship management",
    "e-commerce",
    "digital marketing",
    "search engine optimization",
    "social media marketing",
    "content marketing",
    "email marketing",
    "mobile marketing",
    "influencer marketing",
    "event marketing",
    "brand management",
    "public relations",
    "advertising",
    "market research",
    "consumer behavior",
    "sales management",
    "retail management",
    "wholesale management",
    "logistics management",
    "surgery",
    "radiology",
    "anesthesiology",
    "dermatology",
    "psychiatry",
    "obstetrics",
    "gynecology",
    "pediatrics",
    "orthopedics",
    "cardiology",
    "neurology",
    "gastroenterology",
    "endocrinology",
    "pulmonology",
    "nephrology",
    "urology",
    "rheumatology",
    "otolaryngology",
    "ophthalmology",
    "pathophysiology",
    "epigenetics",
    "molecular biology",
    "cell biology",
    "developmental biology",
    "evolutionary biology",
    "microbiology",
    "virology",
    "parasitology",
    "mycology",
    "entomology",
    "herpetology",
    "ornithology",
    "ichthyology",
    "zoology",
    "botany",
    "horticulture",
    "forestry",
    "agronomy",
    "soil science",
    "hydrology",
    "environmental engineering",
    "sustainability studies",
    "renewable energy",
    "wildlife conservation",
    "natural resource management",
    "climate change studies",
    "disaster management",
    "grade school mathematics",
    "probability",
    "measure theory",
    "real analysis",
    "complex analysis",
    "differential equations",
    "topology",
    "combinatorics",
    "graph theory",
    "cryptography",
    "operating systems",
    "database systems",
    "computer networks",
    "human-computer interaction",
    "compiler design",
    "computer architecture",
    "distributed systems",
    "information theory",
    "signal processing",
    "control theory",
    "thermodynamics",
    "fluid mechanics",
    "optics",
    "solid mechanics",
    "nuclear physics",
    "particle physics",
    "quantum mechanics",
    "statistical mechanics",
    "classical mechanics",
    "organic chemistry",
    "inorganic chemistry",
    "physical chemistry",
    "analytical chemistry",
    "theoretical chemistry",
    "evolution",
    "ecology",
    "cellular biology",
    "molecular genetics",
    "behavioral psychology",
    "social psychology",
    "developmental psychology",
    "industrial-organizational psychology",
    "macroeconomics",
    "microeconomics",
    "international economics",
    "development economics",
    "behavioral economics",
    "constitutional law",
    "criminal law",
    "corporate law",
    "intellectual property law",
    "environmental law",
    "international law",
    "family law",
    "tax law",
    "labor law",
    "mergers and acquisitions",
    "unknown",
]


# Use the FIRST token of each field name
# Use the FIRST token of each field name
ALLOWED_TOKEN_IDS = set()
for field in ALLOWED_FIELDS:
    tok = tokenizer(field, add_special_tokens=False)["input_ids"]
    if len(tok) > 0:
        ALLOWED_TOKEN_IDS.add(tok[0])

LOGITS_PROCESSOR = ClosedSetLogitsProcessor(ALLOWED_TOKEN_IDS)

# -----------------------------
# Sentence-embedding shortlist
# -----------------------------

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SHORTLIST_K = 12  # tune: 8–20

embedder = SentenceTransformer(
    EMBED_MODEL_NAME,
    device="cpu",
)

# Precompute normalized embeddings for fields (once)
field_embeddings = embedder.encode(
    ALLOWED_FIELDS,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True,
)

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

def is_valid_category(cat: str) -> bool:
    if not cat:
        return False
    # Too long to be a field name
    if len(cat.split()) > 4:
        return False
    banned = [
        "write", "response", "answer", "instruction",
        "follow", "process", "unable", "cannot",
        "provide", "classify", "determine"
    ]
    return not any(b in cat for b in banned)

# -----------------------------
# Reasoning Category Prompt
# -----------------------------

CLASSIFICATION_PROMPT = """Classify the QUESTION into the single best matching FIELD from a predefined list.

Rules:
- Classify by subject matter only (not by "type of reasoning").
- Output ONLY the field name (no extra words, no punctuation).
- If none of the predefined fields apply, output: unknown

QUESTION:
{question}

FIELD:"""

# -----------------------------
# Classification Function
# -----------------------------



# --- Efficient closed-set classification via batched log-likelihood scoring ---
def shortlist_fields(question: str):
    """
    Return indices of top-K candidate fields using cosine similarity.
    """
    q_emb = embedder.encode(
        [question],
        normalize_embeddings=True,
    )[0]

    sims = field_embeddings @ q_emb  # cosine since normalized
    topk_idx = np.argpartition(-sims, SHORTLIST_K)[:SHORTLIST_K]
    return topk_idx.tolist()

@torch.no_grad()
def score_fields(questions):
    """
    Efficient closed-set classification via batched log-likelihood scoring.
    """
    device = model.device
    results = []

    for q in questions:
        # --- shortlist fields ---
        candidate_idx = shortlist_fields(q)
        candidate_fields = [ALLOWED_FIELDS[i] for i in candidate_idx]

        # tokenize candidate fields
        field_encodings = [
            tokenizer(" " + f, add_special_tokens=False, return_tensors="pt")
            for f in candidate_fields
        ]

        prompt = CLASSIFICATION_PROMPT.format(question=q)

        enc_prompt = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        prompt_len = enc_prompt.input_ids.shape[1]

        input_ids = []
        attention_masks = []
        field_token_lens = []

        max_field_len = max(fe["input_ids"].shape[1] for fe in field_encodings)

        for fe in field_encodings:
            field_ids = fe["input_ids"].to(device)
            field_len = field_ids.shape[1]

            if field_len < max_field_len:
                pad = torch.full(
                    (1, max_field_len - field_len),
                    tokenizer.pad_token_id,
                    device=device,
                    dtype=field_ids.dtype,
                )
                field_ids = torch.cat([field_ids, pad], dim=1)

            ids = torch.cat([enc_prompt.input_ids, field_ids], dim=1)
            mask = torch.ones_like(ids)

            input_ids.append(ids)
            attention_masks.append(mask)
            field_token_lens.append(field_len)

        scores = []
        num_fields = len(field_encodings)

        for start in range(0, num_fields, FIELD_BATCH_SIZE):
            end = min(start + FIELD_BATCH_SIZE, num_fields)

            batch_input_ids = torch.cat(input_ids[start:end], dim=0)
            batch_attention_masks = torch.cat(attention_masks[start:end], dim=0)
            batch_field_lens = field_token_lens[start:end]

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
            )

            batch_logits = outputs.logits

            for j, field_len in enumerate(batch_field_lens):
                field_logits = batch_logits[
                    j,
                    prompt_len - 1 : prompt_len - 1 + field_len,
                    :
                ]
                log_probs = torch.log_softmax(field_logits, dim=-1)

                field_ids = field_encodings[start + j]["input_ids"].to(device).squeeze(0)

                token_logprobs = log_probs.gather(
                    -1, field_ids.unsqueeze(-1)
                ).squeeze(-1)

                scores.append(token_logprobs.sum().item())

        best_local = int(torch.tensor(scores).argmax())
        results.append(candidate_fields[best_local])

    return results

def classify_reasoning_batch(questions):
    """
    Backwards-compatible wrapper.
    """
    return score_fields(questions)


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
# Multiprocessing worker logic
# -----------------------------

def run_worker(shard_id, num_shards):
    global args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(shard_id)
    torch.cuda.set_device(0)
    args.shard_id = shard_id
    args.num_shards = num_shards

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
    for i, ex in enumerate(tqdm(ds["train"], desc="Classifying examples")):
        if (i % args.num_shards) != args.shard_id:
            continue
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
    output_path = f"axbench/reasoning_data/reasoning_data_shard{args.shard_id}.jsonl"
    with open(output_path, "w") as f:
        for ex in output_rows:
            f.write(json.dumps(ex) + "\n")

    print(f"Done! Wrote shard {args.shard_id}/{args.num_shards} to {output_path}")
    print("Discovered categories:")
    for k, v in concept_map.items():
        print(f"{v}: {k}")


# -----------------------------
# Main entrypoint for multiprocessing
# -----------------------------

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        print(f"Launching {num_gpus} workers (one per GPU)")
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            run_worker,
            args=(num_gpus,),
            nprocs=num_gpus,
            join=True,
        )
    else:
        print("Single GPU detected; running in single-process mode")
        run_worker(0, 1)