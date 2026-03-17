"""
Evaluate three model variants on the test set:
  1. Base model (no fine-tuning)
  2. SFT model (after LoRA fine-tuning)
  3. SFT + DPO model (after preference alignment)

Reports: ROUGE scores, response quality metrics, and comparison table.
Uses the ultrachat_200k test split for evaluation.
"""

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    MODEL_NAME, DATA_DIR, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, USE_NESTED_QUANT,
    EVAL_SAMPLE_SIZE, EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE,
)


def load_test_data() -> list[dict]:
    """Load test samples from disk."""
    test_path = os.path.join(DATA_DIR, "test_samples.json")
    with open(test_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return samples[:EVAL_SAMPLE_SIZE]


def get_bnb_config():
    """Create quantization config."""
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )


def load_model_and_tokenizer(variant: str):
    """
    Load model for a given variant.
    variant: "base", "sft", "dpo"
    """
    print(f"\n>> Loading {variant} model...")
    bnb_config = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if variant == "base":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    elif variant == "sft":
        model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(SFT_OUTPUT_DIR, trust_remote_code=True)
    elif variant == "dpo":
        # First apply SFT adapters, merge, then apply DPO adapters
        model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, DPO_OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(DPO_OUTPUT_DIR, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: list[dict]) -> str:
    """Generate model response for given chat messages."""
    # Build prompt from messages (only system + user, no assistant)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]

    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=EVAL_MAX_NEW_TOKENS,
            temperature=EVAL_TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


def compute_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute simple text overlap metrics between prediction and reference.
    Returns unigram precision, recall, F1 (ROUGE-1 style).
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(overlap) / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_response_quality(prediction: str, reference: str) -> dict[str, float]:
    """
    Compute response quality metrics:
    - Length ratio: how close the response length is to the reference
    - Non-empty: whether the model produced a meaningful response
    - Coherence: basic check that response doesn't repeat excessively
    """
    pred_len = len(prediction.split())
    ref_len = len(reference.split())

    # Length ratio (1.0 = same length, <1 = shorter, >1 = longer)
    length_ratio = min(pred_len / max(ref_len, 1), 2.0)

    # Non-empty check
    is_non_empty = 1.0 if pred_len > 5 else 0.0

    # Repetition check (ratio of unique tokens)
    tokens = prediction.lower().split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)

    return {
        "length_ratio": length_ratio,
        "non_empty": is_non_empty,
        "unique_ratio": unique_ratio,
    }


def evaluate_variant(model, tokenizer, test_data: list[dict], variant_name: str) -> dict:
    """Run evaluation on all test samples for one model variant."""
    total = min(len(test_data), EVAL_SAMPLE_SIZE)
    metrics_accum = {
        "rouge_precision": 0.0,
        "rouge_recall": 0.0,
        "rouge_f1": 0.0,
        "length_ratio": 0.0,
        "non_empty": 0.0,
        "unique_ratio": 0.0,
    }

    print(f"\n{'='*50}")
    print(f"  Evaluating: {variant_name} ({total} samples)")
    print(f"{'='*50}")

    for i, sample in enumerate(test_data[:total]):
        # Use chat messages if available, otherwise build from input
        if "messages" in sample:
            messages = sample["messages"]
        else:
            messages = [
                {"role": "user", "content": sample["input"]},
            ]

        prediction = generate_response(model, tokenizer, messages)
        reference = sample.get("expected_output", "")

        # Compute metrics
        rouge = compute_rouge_scores(prediction, reference)
        quality = compute_response_quality(prediction, reference)

        metrics_accum["rouge_precision"] += rouge["precision"]
        metrics_accum["rouge_recall"] += rouge["recall"]
        metrics_accum["rouge_f1"] += rouge["f1"]
        metrics_accum["length_ratio"] += quality["length_ratio"]
        metrics_accum["non_empty"] += quality["non_empty"]
        metrics_accum["unique_ratio"] += quality["unique_ratio"]

        if (i + 1) % 10 == 0 or i == 0:
            avg_f1 = metrics_accum["rouge_f1"] / (i + 1)
            avg_nonempty = metrics_accum["non_empty"] / (i + 1)
            print(f"   [{i+1:3d}/{total}] "
                  f"ROUGE-1 F1: {avg_f1:.3f}  |  Non-empty: {avg_nonempty:.1%}")

    # Average all metrics
    result = {k: v / total for k, v in metrics_accum.items()}
    return result


def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON RESULTS (Before -> After)")
    print("=" * 72)

    header = f"{'Metric':<25}"
    for name in all_metrics:
        header += f" {name:>12}"
    print(header)
    print("-" * 72)

    metric_labels = {
        "rouge_f1":       "ROUGE-1 F1",
        "rouge_precision": "ROUGE-1 Precision",
        "rouge_recall":    "ROUGE-1 Recall",
        "non_empty":       "Non-Empty Rate",
        "length_ratio":    "Length Ratio",
        "unique_ratio":    "Unique Token Ratio",
    }

    for metric_key, label in metric_labels.items():
        row = f"  {label:<23}"
        for name, metrics in all_metrics.items():
            val = metrics.get(metric_key, 0)
            if metric_key == "length_ratio":
                row += f" {val:>11.2f}x"
            else:
                row += f" {val:>11.1%}"
        print(row)

    print("=" * 72)

    # Show improvement summary
    names = list(all_metrics.keys())
    if len(names) >= 2:
        base_f1 = all_metrics[names[0]]["rouge_f1"]
        sft_f1 = all_metrics[names[1]]["rouge_f1"]
        delta = sft_f1 - base_f1
        print(f"\n  => SFT vs Base:  ROUGE-1 F1  {base_f1:.1%} -> {sft_f1:.1%}  "
              f"({'+'if delta>=0 else ''}{delta:.1%})")
    if len(names) >= 3:
        sft_f1 = all_metrics[names[1]]["rouge_f1"]
        dpo_f1 = all_metrics[names[2]]["rouge_f1"]
        delta = dpo_f1 - sft_f1
        print(f"  => DPO vs SFT:   ROUGE-1 F1  {sft_f1:.1%} -> {dpo_f1:.1%}  "
              f"({'+'if delta>=0 else ''}{delta:.1%})")


def main():
    print("=" * 60)
    print("  Model Evaluation: Base -> SFT -> SFT+DPO")
    print("=" * 60)

    test_data = load_test_data()
    print(f"\n>> Loaded {len(test_data)} test samples")

    all_metrics = {}

    # Determine which variants to evaluate
    variants = ["base"]
    if os.path.exists(SFT_OUTPUT_DIR):
        variants.append("sft")
    else:
        print(f"\n  [!] SFT model not found at {SFT_OUTPUT_DIR}. Skipping.")

    if os.path.exists(DPO_OUTPUT_DIR):
        variants.append("dpo")
    else:
        print(f"\n  [!] DPO model not found at {DPO_OUTPUT_DIR}. Skipping.")

    for variant in variants:
        model, tokenizer = load_model_and_tokenizer(variant)
        metrics = evaluate_variant(model, tokenizer, test_data, variant.upper())
        all_metrics[variant.upper()] = metrics

        # Free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Print results table
    print_comparison_table(all_metrics)

    # Save results to JSON
    results_path = os.path.join(DATA_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n>> Results saved to {results_path}")


if __name__ == "__main__":
    main()
