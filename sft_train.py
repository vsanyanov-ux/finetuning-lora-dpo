"""
SFT (Supervised Fine-Tuning) with QLoRA.

Loads the base model with 4-bit quantization, attaches LoRA adapters,
and trains on the JSON-extraction SFT dataset.
"""

import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from config import (
    MODEL_NAME, DATA_DIR, SFT_OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, USE_NESTED_QUANT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    SFT_EPOCHS, SFT_BATCH_SIZE, SFT_GRADIENT_ACCUMULATION,
    SFT_LEARNING_RATE, SFT_MAX_SEQ_LENGTH, SFT_WARMUP_RATIO,
    SFT_LOGGING_STEPS, SFT_SAVE_STEPS,
)


def create_bnb_config():
    """Create BitsAndBytes quantization config for 4-bit loading."""
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )


def create_lora_config():
    """Create LoRA configuration."""
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main():
    print("=" * 60)
    print("  SFT Fine-Tuning with QLoRA")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────────
    sft_path = os.path.join(DATA_DIR, "sft_dataset")
    print(f"\n>> Loading SFT dataset from {sft_path}...")
    dataset = load_from_disk(sft_path)
    # Remove extra columns that might confuse SFTTrainer
    cols_to_remove = [c for c in dataset["train"].column_names if c != "messages"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    print(f"   Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

    # ── Load tokenizer ───────────────────────────────────────────────────
    print(f"\n>> Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load model with quantization ─────────────────────────────────────
    print(f"\n>> Loading model with 4-bit quantization: {MODEL_NAME}")
    bnb_config = create_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    # ── Prepare model for k-bit training ─────────────────────────────────
    model = prepare_model_for_kbit_training(model)

    # ── Apply LoRA ───────────────────────────────────────────────────────
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\n[INFO] Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── Training arguments ───────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=SFT_OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRADIENT_ACCUMULATION,
        learning_rate=SFT_LEARNING_RATE,
        max_length=SFT_MAX_SEQ_LENGTH,
        warmup_ratio=SFT_WARMUP_RATIO,
        logging_steps=SFT_LOGGING_STEPS,
        save_steps=SFT_SAVE_STEPS,
        save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        eval_strategy="steps",
        eval_steps=SFT_SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    print("\n>> Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n>> Saving model to {SFT_OUTPUT_DIR}...")
    trainer.save_model(SFT_OUTPUT_DIR)
    tokenizer.save_pretrained(SFT_OUTPUT_DIR)

    print("\nDONE: SFT training complete!")
    print(f"   Model saved to: {SFT_OUTPUT_DIR}")

    # ── Log final metrics ────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print(f"\n[INFO] Final eval loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
