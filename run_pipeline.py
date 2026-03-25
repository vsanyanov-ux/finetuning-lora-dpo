"""
Complete fine-tuning pipeline runner.

Run the entire pipeline end-to-end:
  1. Prepare datasets
  2. SFT training with QLoRA
  3. DPO alignment training
  4. Evaluate all model variants

Usage:
  python run_pipeline.py            # Run full pipeline
  python run_pipeline.py --step 1   # Run only dataset preparation
  python run_pipeline.py --step 2   # Run only SFT training
  python run_pipeline.py --step 3   # Run only DPO training
  python run_pipeline.py --step 4   # Run only evaluation
"""

import argparse
import subprocess
import sys
import time

STEPS = {
    1: ("Dataset Preparation", "prepare_dataset.py"),
    2: ("SFT Training (QLoRA)", "sft_train.py"),
    3: ("DPO Alignment", "dpo_train.py"),
    4: ("Evaluation", "evaluate_model.py"),
}


def run_step(step_num: int):
    """Run a pipeline step."""
    name, script = STEPS[step_num]
    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}/4: {name}")
    print(f"  Script: {script}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - start

    print(f"\n[DONE] Step {step_num} ({name}) completed in {elapsed:.1f}s")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Fine-Tuning Pipeline Runner")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Run only a specific step (1-4)")
    args = parser.parse_args()

    print(">> Fine-Tuning with LoRA & DPO Pipeline")
    print("=" * 60)

    if args.step:
        run_step(args.step)
    else:
        total_start = time.time()
        for step_num in STEPS:
            run_step(step_num)
        total_elapsed = time.time() - total_start
        print(f"\n[FINISH] Full pipeline completed in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
