import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import (
    CPOTrainer,
    ModelConfig,
    CPOConfig,
)
from peft import LoraConfig, get_peft_model
import logging
logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

os.environ["WANDB_PROJECT"] = "Qwen3-1.7B-TLDR"
os.environ["WANDB_RUN_NAME"] = "cpo_config2"
os.environ["WANDB_LOG_MODEL"] = "false"

MODEL_PATH = "../model/Qwen3-1.7B-sft-config3"
OUTPUT_DIR = "../model/Qwen3-1.7B-cpo-config2"

TRAIN_DATA = "../data/TLDR/dpo/train_conversational.jsonl"
TEST_DATA = "../data/TLDR/dpo/test_conversational.jsonl"

def main():
    logger.info("Starting SimPO training")

    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "test": TEST_DATA})
    logger.info(f"Dataset loaded: {dataset}")
    logger.info(f"First training example: {dataset['train'][0]}")
    logger.info(f"First test example: {dataset['test'][0]}")

    logger.info(f"Loading model from {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_args = ModelConfig(
        model_name_or_path=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model_kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_kwargs["trust_remote_code"],
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_kwargs["trust_remote_code"],
    )

    logger.info("Model loaded")
    
    logger.info(f"configure simpo_config")

    simpo_config = CPOConfig(
        loss_type="simpo",
        beta=0.1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        output_dir=OUTPUT_DIR,
        save_steps=500,
        save_safetensors=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        report_to="wandb",
    )

    trainer = CPOTrainer(
        model=model,
        args=simpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    logger.info("Trainer initialized")
    logger.info("Starting training")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logger.info("Model saved")

if __name__ == "__main__":
    main()