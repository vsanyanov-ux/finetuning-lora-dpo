"""
DPO (Direct Preference Optimization) alignment training.

Takes the SFT-tuned model and further aligns it using preference pairs
(chosen vs rejected completions) to improve output quality.
"""

import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer

from config import (
    MODEL_NAME, DATA_DIR, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, USE_NESTED_QUANT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    DPO_EPOCHS, DPO_BATCH_SIZE, DPO_GRADIENT_ACCUMULATION,
    DPO_LEARNING_RATE, DPO_MAX_LENGTH, DPO_MAX_PROMPT_LENGTH,
    DPO_BETA, DPO_WARMUP_RATIO, DPO_LOGGING_STEPS, DPO_SAVE_STEPS,
)


def main():
    print("=" * 60)
    print("  DPO Alignment Training")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────────
    dpo_path = os.path.join(DATA_DIR, "dpo_dataset")
    print(f"\n>> Loading DPO dataset from {dpo_path}...")
    dataset = load_from_disk(dpo_path)
    print(f"   Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # ── Load tokenizer ───────────────────────────────────────────────────
    print(f"\n>> Loading tokenizer from SFT checkpoint: {SFT_OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(SFT_OUTPUT_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO requires left-padding

    # ── Load base model with quantization ────────────────────────────────
    print(f"\n>> Loading base model with 4-bit quantization: {MODEL_NAME}")
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── Load SFT LoRA adapters ───────────────────────────────────────────
    abs_sft_path = os.path.abspath(SFT_OUTPUT_DIR)
    print(f"\n>> Loading SFT LoRA adapters from: {abs_sft_path}")
    
    if not os.path.exists(abs_sft_path):
        raise FileNotFoundError(
            f"ERROR: SFT results not found at {abs_sft_path}. "
            f"Did Step 2 (sft_train.py) complete successfully?"
        )
    
    # Check for essential PEFT file
    config_file = os.path.join(abs_sft_path, "adapter_config.json")
    if not os.path.exists(config_file):
        files = os.listdir(abs_sft_path)
        raise FileNotFoundError(
            f"ERROR: 'adapter_config.json' not found in {abs_sft_path}.\n"
            f"Directory contains: {files}\n"
            f"This usually means SFT training was interrupted and did not save the model."
        )

    model = PeftModel.from_pretrained(model, abs_sft_path)
    model = model.merge_and_unload()  # Merge SFT adapters into the base model

    # ── New LoRA config for DPO ──────────────────────────────────────────
    dpo_lora_config = LoraConfig(
        r=LORA_R // 2,                       # Smaller rank for DPO
        lora_alpha=LORA_ALPHA // 2,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── DPO Training ─────────────────────────────────────────────────────
    print("\n>> Configuring DPO training...")
    training_args = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        num_train_epochs=DPO_EPOCHS,
        per_device_train_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRADIENT_ACCUMULATION,
        learning_rate=DPO_LEARNING_RATE,
        max_length=DPO_MAX_LENGTH,
        beta=DPO_BETA,
        warmup_ratio=DPO_WARMUP_RATIO,
        logging_steps=DPO_LOGGING_STEPS,
        save_steps=DPO_SAVE_STEPS,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        eval_strategy="no",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    print("\n>> Starting DPO training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=dpo_lora_config,
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n>> Saving DPO model to {DPO_OUTPUT_DIR}...")
    trainer.save_model(DPO_OUTPUT_DIR)
    tokenizer.save_pretrained(DPO_OUTPUT_DIR)

    print("\nDONE: DPO training complete!")
    print(f"   Model saved to: {DPO_OUTPUT_DIR}")

    # ── Log final metrics ────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print(f"\n[INFO] Final eval metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
