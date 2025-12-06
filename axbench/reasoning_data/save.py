import os
from datasets import load_dataset

INPUT_JSONL = "axbench/reasoning_data/reasoning_data.jsonl"
OUTPUT_DIR = "axbench/reasoning_data/reasoning_hf_parquet"  # directory to hold parquet files

def main():
    # 1. Load the JSONL as a HF dataset
    ds = load_dataset(
        "json",
        data_files={"train": INPUT_JSONL},
        split="train",
    )

    # 2. Filter by output_concept
    # Keep only entries whose output_concept contains one of the target words
    target_substrings = ["analysis", "explanation", "reasoning"]

    def keep_example(ex):
        concept = (ex.get("output_concept") or "").lower()
        return any(key in concept for key in target_substrings)

    ds_filtered = ds.filter(keep_example)

    print(f"Original size: {len(ds)}")
    print(f"Filtered size: {len(ds_filtered)}")

    # 3. Make output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Save as parquet files (Hugging Face-friendly)
    # This will create one parquet file per shard inside OUTPUT_DIR
    ds_filtered.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"))

    # Optionally, also save as a HF dataset directory for local inspection
    # ds_filtered.save_to_disk("reasoning_hf_dataset")

    print(f"Wrote filtered parquet dataset to: {OUTPUT_DIR}/train.parquet")

if __name__ == "__main__":
    main()