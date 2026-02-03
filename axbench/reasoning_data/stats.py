from datasets import load_dataset
from collections import Counter
import itertools

# Load dataset (all splits if present)
ds = load_dataset("vennemeyerd/axbench-reasoning")

# Some HF datasets have multiple splits; flatten them
def iter_rows(dataset_dict):
    for split in dataset_dict.values():
        for row in split:
            yield row

from datasets import Dataset

# Build a balanced subset:
# - first 16,000 examples
# - no output_concept appears more than 20 times
max_total = 16_000
max_per_concept = 20

concept_counter = Counter()
kept_rows = []

for row in iter_rows(ds):
    if len(kept_rows) >= max_total:
        break

    concept = row.get("output_concept")
    if concept is None or concept == "":
        continue

    if concept_counter[concept] >= max_per_concept:
        continue

    kept_rows.append(row)
    concept_counter[concept] += 1

print(f"Collected {len(kept_rows)} examples")
print(f"Unique concepts: {len(concept_counter)}")

# Create and save dataset
balanced_ds = Dataset.from_list(kept_rows)
balanced_ds.save_to_disk("balanced_16k_no_over20")