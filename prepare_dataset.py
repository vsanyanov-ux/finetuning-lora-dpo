"""
Dataset preparation for the fine-tuning pipeline.

Downloads recommended datasets from HuggingFace:
  - SFT:  HuggingFaceH4/ultrachat_200k   (multi-turn chat)
  - DPO:  argilla/distilabel-intel-orca-dpo-pairs  (preference pairs)

Formats and saves processed datasets to DATA_DIR.
"""

import json
import os

from datasets import Dataset, DatasetDict, load_dataset

from config import DATA_DIR, EVAL_SAMPLE_SIZE

# ── Seed ────────────────────────────────────────────────────────────────
SEED = 42

# ── Dataset configs ─────────────────────────────────────────────────────
SFT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
SFT_SUBSET = "default"
SFT_TRAIN_SIZE = 1000       # subset for efficient training
SFT_VAL_SIZE = 100
SFT_TEST_SIZE = EVAL_SAMPLE_SIZE

DPO_DATASET_NAME = "argilla/distilabel-intel-orca-dpo-pairs"
DPO_TRAIN_SIZE = 1000
DPO_VAL_SIZE = 100

# ── System prompt for JSON extraction fine-tuning ───────────────────────
SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instructions carefully "
    "and provide accurate, well-structured responses."
)


def prepare_sft_dataset() -> DatasetDict:
    """
    Load and format the ultrachat_200k dataset for SFT.
    The dataset already has 'messages' field in chat format.
    """
    print(f"\n>> Loading SFT dataset: {SFT_DATASET_NAME}")

    # Load train and test splits
    ds_train = load_dataset(SFT_DATASET_NAME, split="train_sft")
    ds_test = load_dataset(SFT_DATASET_NAME, split="test_sft")

    print(f"   Full train size: {len(ds_train):,}")
    print(f"   Full test size:  {len(ds_test):,}")

    # Shuffle and take subsets
    ds_train = ds_train.shuffle(seed=SEED).select(range(min(SFT_TRAIN_SIZE + SFT_VAL_SIZE, len(ds_train))))
    ds_test = ds_test.shuffle(seed=SEED).select(range(min(SFT_TEST_SIZE, len(ds_test))))

    # Split train into train + validation
    train_val = ds_train.train_test_split(test_size=SFT_VAL_SIZE, seed=SEED)

    sft_dataset = DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": ds_test,
    })

    print(f"   Selected -> Train: {len(sft_dataset['train'])}, "
          f"Val: {len(sft_dataset['validation'])}, "
          f"Test: {len(sft_dataset['test'])}")

    return sft_dataset


def prepare_dpo_dataset() -> DatasetDict:
    """
    Load and format the distilabel-intel-orca-dpo-pairs for DPO.
    Format each sample into: prompt, chosen, rejected.
    """
    print(f"\n>> Loading DPO dataset: {DPO_DATASET_NAME}")

    ds = load_dataset(DPO_DATASET_NAME, split="train")
    print(f"   Full dataset size: {len(ds):,}")

    # Filter to high-quality samples (status == 'chosen' indicates valid pair)
    ds = ds.filter(lambda x: x.get("status") == "tie" or x.get("status") is not None)
    print(f"   After filtering: {len(ds):,}")

    # Shuffle and take subset
    ds = ds.shuffle(seed=SEED).select(range(min(DPO_TRAIN_SIZE + DPO_VAL_SIZE, len(ds))))

    # Format for DPO trainer
    def format_dpo_sample(example):
        """Convert raw DPO sample to chat format."""
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
        ]

        # chosen and rejected are the full text responses
        chosen = [{"role": "assistant", "content": example.get("chosen", "")}]
        rejected = [{"role": "assistant", "content": example.get("rejected", "")}]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    ds = ds.map(format_dpo_sample, remove_columns=ds.column_names)

    # Split into train + validation
    split = ds.train_test_split(test_size=DPO_VAL_SIZE, seed=SEED)

    dpo_dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    print(f"   Selected -> Train: {len(dpo_dataset['train'])}, "
          f"Val: {len(dpo_dataset['validation'])}")

    return dpo_dataset


def save_test_samples(sft_dataset: DatasetDict):
    """Save test samples as JSON for the evaluation script."""
    test_samples = []
    for sample in sft_dataset["test"]:
        messages = sample["messages"]
        # Extract the user message and assistant response
        user_msg = ""
        assistant_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]

        if user_msg and assistant_msg:
            test_samples.append({
                "input": user_msg,
                "expected_output": assistant_msg,
                "messages": messages,
            })

    test_path = os.path.join(DATA_DIR, "test_samples.json")
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"\n   >> Test samples saved to {test_path} ({len(test_samples)} samples)")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  Dataset Preparation")
    print("  SFT:  HuggingFaceH4/ultrachat_200k")
    print("  DPO:  argilla/distilabel-intel-orca-dpo-pairs")
    print("=" * 60)

    # ── SFT dataset ──────────────────────────────────────────────────────
    sft_dataset = prepare_sft_dataset()
    sft_path = os.path.join(DATA_DIR, "sft_dataset")
    sft_dataset.save_to_disk(sft_path)
    print(f"   >> SFT dataset saved to {sft_path}")

    # ── DPO dataset ──────────────────────────────────────────────────────
    dpo_dataset = prepare_dpo_dataset()
    dpo_path = os.path.join(DATA_DIR, "dpo_dataset")
    dpo_dataset.save_to_disk(dpo_path)
    print(f"   >> DPO dataset saved to {dpo_path}")

    # ── Save test samples for evaluation ────────────────────────────────
    save_test_samples(sft_dataset)

    print("\n" + "=" * 60)
    print("  Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
