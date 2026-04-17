import subprocess
import json
import os
from config import EVAL_SAMPLE_SIZE, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR, DATA_DIR

def run_variant(variant):
    print(f"\n[ORCHESTRATOR] >>> Starting full evaluation for {variant.upper()}...")
    try:
        # Pass samples to match config
        subprocess.run(["python", "evaluate_single.py", "--variant", variant, "--samples", str(EVAL_SAMPLE_SIZE)], check=True)
        print(f"[ORCHESTRATOR] {variant.upper()} finished successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"[ORCHESTRATOR] {variant.upper()} FAILED.")
        return False

def print_comparison_table(all_metrics: dict):
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON RESULTS (Full Evaluation)")
    print("=" * 72)
    header = f"{'Metric':<25}"
    for name in all_metrics:
        header += f" {name:>12}"
    print(header)
    print("-" * 72)
    metric_labels = {
        "rouge_f1": "ROUGE-1 F1",
        "rouge_precision": "ROUGE-1 Precision",
        "rouge_recall": "ROUGE-1 Recall",
        "non_empty": "Non-Empty Rate",
        "length_ratio": "Length Ratio",
        "unique_ratio": "Unique Token Ratio",
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

def main():
    variants = ["base"]
    if os.path.exists(SFT_OUTPUT_DIR):
        variants.append("sft")
    if os.path.exists(DPO_OUTPUT_DIR):
        variants.append("dpo")
    
    all_metrics = {}
    
    for v in variants:
        if run_variant(v):
            temp_path = f"temp_res_{v}.json"
            if os.path.exists(temp_path):
                with open(temp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_metrics[v.upper()] = data["metrics"]
    
    # Print table
    if all_metrics:
        print_comparison_table(all_metrics)
        
        # Save results
        results_path = os.path.join(DATA_DIR, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n>> All results saved to {results_path}")
    else:
        print("\n[!] No results to display.")

if __name__ == "__main__":
    main()
