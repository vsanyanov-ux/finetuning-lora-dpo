import json
import os
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import (
    MODEL_NAME, DATA_DIR, SFT_OUTPUT_DIR, DPO_OUTPUT_DIR,
    LOAD_IN_4BIT, BNB_4BIT_QUANT_TYPE, BNB_4BIT_COMPUTE_DTYPE, USE_NESTED_QUANT,
    EVAL_MAX_NEW_TOKENS, EVAL_TEMPERATURE, EVAL_BATCH_SIZE
)

def get_bnb_config():
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    return BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

def load_variant(variant: str):
    print(f"\n>> [Subprocess] Loading {variant.upper()} model...", flush=True)
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
        model = PeftModel.from_pretrained(model, SFT_OUTPUT_DIR)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, DPO_OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(DPO_OUTPUT_DIR, trust_remote_code=True)
    
    # Critical for decoder-only batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def compute_rouge_scores(prediction: str, reference: str) -> dict[str, float]:
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
    pred_len = len(prediction.split())
    ref_len = len(reference.split())
    length_ratio = min(pred_len / max(ref_len, 1), 2.0)
    is_non_empty = 1.0 if pred_len > 5 else 0.0
    tokens = prediction.lower().split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    return {
        "length_ratio": length_ratio,
        "non_empty": is_non_empty,
        "unique_ratio": unique_ratio,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    test_path = os.path.join(DATA_DIR, "test_samples.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)[:args.samples]

    model, tokenizer = load_variant(args.variant)
    model.eval()

    total = len(test_data)
    metrics_accum = {
        "rouge_precision": 0.0, "rouge_recall": 0.0, "rouge_f1": 0.0,
        "length_ratio": 0.0, "non_empty": 0.0, "unique_ratio": 0.0,
    }

    outputs = []
    print(f">> [Subprocess] Evaluating {args.variant.upper()} ({total} samples, batch_size={EVAL_BATCH_SIZE})...", flush=True)
    
    # Process in batches
    for i in range(0, total, EVAL_BATCH_SIZE):
        batch = test_data[i : i + EVAL_BATCH_SIZE]
        prompts = []
        references = []
        
        for sample in batch:
            messages = sample.get("messages", [{"role": "user", "content": sample["input"]}])
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            references.append(sample.get("expected_output", ""))
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=EVAL_MAX_NEW_TOKENS,
                temperature=EVAL_TEMPERATURE,
                do_sample=EVAL_TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode and compute metrics for the batch
        for j, (out_ids, ref) in enumerate(zip(gen_out, references)):
            # Skip input tokens
            input_len = inputs["input_ids"].shape[1]
            pred_ids = out_ids[input_len:]
            prediction = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            
            # Compute metrics
            rouge = compute_rouge_scores(prediction, ref)
            quality = compute_response_quality(prediction, ref)

            metrics_accum["rouge_precision"] += rouge["precision"]
            metrics_accum["rouge_recall"] += rouge["recall"]
            metrics_accum["rouge_f1"] += rouge["f1"]
            metrics_accum["length_ratio"] += quality["length_ratio"]
            metrics_accum["non_empty"] += quality["non_empty"]
            metrics_accum["unique_ratio"] += quality["unique_ratio"]
            
            outputs.append(prediction)
        
        # Progress reporting
        processed = min(i + EVAL_BATCH_SIZE, total)
        avg_f1 = metrics_accum["rouge_f1"] / processed
        print(f"   [{processed:3d}/{total}] Avg ROUGE-1 F1: {avg_f1:.3f}", flush=True)

    # Average and save
    result = {
        "variant": args.variant,
        "metrics": {k: v / total for k, v in metrics_accum.items()},
        "outputs": outputs
    }
    
    temp_path = f"temp_res_{args.variant}.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f">> [Subprocess] Results saved to {temp_path}", flush=True)

if __name__ == "__main__":
    main()
