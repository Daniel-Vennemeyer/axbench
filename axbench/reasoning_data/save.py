from datasets import load_dataset
import os

INPUT_JSONL = "axbench/reasoning_data/reasoning_data.jsonl"
HF_REPO_ID = "vennemeyerd/axbench-reasoning"
HF_SPLIT = "train"

def main():
    # 1. Load the JSONL as a HF dataset
    ds = load_dataset(
        "json",
        data_files={HF_SPLIT: INPUT_JSONL},
        split=HF_SPLIT,
    )

    # 2. Filter by output_concept
    target_substrings = ["analysis", "explanation", "reasoning"]

    def keep_example(ex):
        concept = (ex.get("output_concept") or "").lower()
        return any(key in concept for key in target_substrings)

    ds_filtered = ds.filter(keep_example)

    print(f"Original size: {len(ds)}")
    print(f"Filtered size: {len(ds_filtered)}")

    # 3. Push directly to Hugging Face Hub (public by default)
    # Assumes you are logged in via `huggingface-cli login`
    ds_filtered.push_to_hub(
        HF_REPO_ID,
        split=HF_SPLIT,
        private=False,
    )

    print(f"Pushed dataset to: https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()