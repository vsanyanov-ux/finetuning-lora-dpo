"""
Central configuration for the fine-tuning pipeline.
All hyperparameters and paths are defined here.
"""

# ── Model ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
SFT_OUTPUT_DIR = "./results/sft-lora"
DPO_OUTPUT_DIR = "./results/dpo-lora"

# ── QLoRA / BitsAndBytes ────────────────────────────────────────────────
LOAD_IN_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
USE_NESTED_QUANT = False

# ── LoRA ─────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── SFT Training ────────────────────────────────────────────────────────
SFT_EPOCHS = 3
SFT_BATCH_SIZE = 4
SFT_GRADIENT_ACCUMULATION = 4
SFT_LEARNING_RATE = 2e-4
SFT_MAX_SEQ_LENGTH = 512
SFT_WARMUP_RATIO = 0.03
SFT_LOGGING_STEPS = 10
SFT_SAVE_STEPS = 50

# ── DPO Training ────────────────────────────────────────────────────────
DPO_EPOCHS = 1
DPO_BATCH_SIZE = 4
DPO_GRADIENT_ACCUMULATION = 4
DPO_LEARNING_RATE = 5e-5
DPO_MAX_LENGTH = 512
DPO_MAX_PROMPT_LENGTH = 256
DPO_BETA = 0.1                 # KL penalty coefficient
DPO_WARMUP_RATIO = 0.1
DPO_LOGGING_STEPS = 10
DPO_SAVE_STEPS = 50

# ── Evaluation ──────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 100          # number of test samples to evaluate
EVAL_MAX_NEW_TOKENS = 256
EVAL_TEMPERATURE = 0.1
