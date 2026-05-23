#!/usr/bin/env python3
"""
exp019_expert_sft_v2: second-stage SFT on format-fixed expert trajectories.

Input:  train.jsonl / val.jsonl (messages array per line, same format as sft_train.py)
Base:   exp008_sft checkpoint (format_ok=100%, action_is_valid=90%)
Output: checkpoint at ./checkpoints/exp019_expert_sft_v2/final/
Key:    LR=5e-6 (4x lower than exp017), epochs=2, to preserve format quality.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODEL_PATH = "./checkpoints/exp008_sft/final"
DEFAULT_TRAIN_PATH = "/data/expert_sft_v3/expert_sft_train.jsonl"
DEFAULT_VAL_PATH = "/data/expert_sft_v3/expert_sft_val.jsonl"
DEFAULT_OUTPUT_DIR = "./checkpoints/exp019b_expert_sft_v3"

SIMPLE_CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


def main():
    parser = argparse.ArgumentParser(description="exp019: second-stage SFT from exp008_sft checkpoint")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--data_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--val_path", type=str, default=DEFAULT_VAL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.chat_template = SIMPLE_CHATML
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    print(f"Loading train dataset from {args.data_path}...")
    train_dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"Train dataset size: {len(train_dataset)}")

    eval_dataset = None
    try:
        eval_dataset = load_dataset("json", data_files=args.val_path, split="train")
        print(f"Val dataset size: {len(eval_dataset)}")
    except Exception as e:
        print(f"No val dataset loaded ({e}), skipping eval.")

    sample = train_dataset[0]["messages"]
    encoded = tokenizer.apply_chat_template(sample, tokenize=True)
    print(f"Sample token length: {len(encoded)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        max_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=1 if args.max_steps > 0 else 5,
        save_strategy="epoch" if args.max_steps < 0 else "no",
        eval_strategy="epoch" if (args.max_steps < 0 and eval_dataset is not None) else "no",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    result = trainer.train()
    print(f"\nTraining complete. Metrics: {result.metrics}")

    if args.max_steps < 0:
        final_path = args.output_dir + "/final"
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
