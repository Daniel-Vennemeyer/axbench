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

CLASSIFICATION_PROMPT = """Question:
{question}

The academic field of this question is:"""


# -----------------------------
# Score-based field classifier
# -----------------------------

@torch.no_grad()
def score_fields(questions):
    """
    Return the best field for each question by computing
    the conditional log-likelihood of each field label.
    """
    device = model.device
    results = []

    for q in questions:
        prompt = CLASSIFICATION_PROMPT.format(question=q)

        # Encode prompt once
        enc_prompt = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        best_field = "unknown"
        best_score = None

        for field in ALLOWED_FIELDS:
            # Score P(field | prompt)
            enc_field = tokenizer(
                " " + field,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)

            input_ids = torch.cat(
                [enc_prompt.input_ids, enc_field.input_ids], dim=1
            )
            attention_mask = torch.ones_like(input_ids)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits[:, -enc_field.input_ids.shape[1]-1:-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            token_logprobs = log_probs.gather(
                -1, enc_field.input_ids.unsqueeze(-1)
            ).squeeze(-1)

            score = token_logprobs.sum().item()

            if best_score is None or score > best_score:
                best_score = score
                best_field = field

        results.append(best_field)

    return results

# -----------------------------
# Classification Function
# -----------------------------

def classify_reasoning_batch(questions):
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