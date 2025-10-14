import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import (
    SFTTrainer,
    ModelConfig,
    SFTConfig,
) 
from peft import LoraConfig, get_peft_model

os.environ["WANDB_PROJECT"] = "Qwen3-1.7B-TLDR"  
os.environ["WANDB_RUN_NAME"] = "sft_config1"
os.environ["WANDB_LOG_MODEL"] = "checkpoint" 

MODEL_PATH = "../model/Qwen3-1.7B"
OUTPUT_DIR = "../model/Qwen3-1.7B-sft"

TRAIN_DATA = "../data/TLDR/sft/train_conversational.jsonl"
TEST_DATA = "../data/TLDR/sft/test_conversational.jsonl"

def main():
    print("=" * 80)
    print("Starting SFT training")
    print("=" * 80)
    
    print("Loading dataset")
    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_DATA,
            "test": TEST_DATA
        }
    )
    
    print(dataset)
    print(dataset['train'][0])
    print(dataset['test'][0])
    
    
    print(f"\n🤖 加载模型: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )
    
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
        dtype=model_args.dtype
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    print(f"✅ model loaded successfully")
    
 
    print(f"configure sftconfig")
    
    sft_config = SFTConfig(
        bf16=True,
        completion_only_loss=False,
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=16,  # 每个GPU的batch size
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        output_dir=OUTPUT_DIR,
        save_steps=500,
        ddp_find_unused_parameters=False,  # DDP优化
        dataloader_num_workers=4,  # 数据加载并行
        report_to="wandb",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    print(f"✅ Trainer created successfully")
    
    print("\n" + "=" * 80)
    print("Start training")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Saving model")
    print("=" * 80 + "\n")
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

    print("\n" + "=" * 80)
    print("✅ 训练完成！")
    print("=" * 80)
   