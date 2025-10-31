import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, ModelConfig, DPOConfig
import logging

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

os.environ["WANDB_PROJECT"] = "Qwen3-1.7B-TLDR"
os.environ["WANDB_RUN_NAME"] = "dpo_aggressive"
os.environ["WANDB_LOG_MODEL"] = "false"

MODEL_PATH = "../model/Qwen3-1.7B-sft-config3"
OUTPUT_DIR = "../model/Qwen3-1.7B-dpo-aggressive"

TRAIN_DATA = "../data/TLDR/dpo/train_conversational.jsonl"
TEST_DATA = "../data/TLDR/dpo/test_conversational.jsonl"

def main():
    logger.info("Starting DPO training - AGGRESSIVE CONFIG")
    
    dataset = load_dataset("json", data_files={"train": TRAIN_DATA, "test": TEST_DATA})
    logger.info(f"Dataset loaded: {dataset}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_args = ModelConfig(
        model_name_or_path=MODEL_PATH,
        dtype="bfloat16",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
    )
    
    logger.info("Models loaded")
    
    # 激进配置：更快收敛，但需要密切监控
    dpo_config = DPOConfig(
        # 损失函数 - 更小的约束
        loss_type="sigmoid",
        beta=0.05,  # 更小的beta，更激进
        
        # 学习率 - 更高但仍在安全范围
        learning_rate=2e-6,
        lr_scheduler_type="linear",  # linear衰减
        warmup_ratio=0.05,  # 最小warmup
        
        # 批次大小 - 更大的per_device batch
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,  # effective_batch=8×8×4=256
        
        # 训练周期
        num_train_epochs=1,
        
        # 评估和保存 - 更频繁评估以监控稳定性
        eval_strategy="steps",
        eval_steps=100,  # 更频繁
        save_strategy="steps",
        save_steps=500,
        save_total_limit=5,  # 保存更多checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 优化器
        optim="adamw_torch",
        weight_decay=0.005,  # 更小的weight decay
        max_grad_norm=1.0,
        
        # 精度
        bf16=True,
        gradient_checkpointing=False,  # 禁用以加快速度
        
        # 日志
        logging_steps=5,  # 更频繁的日志
        output_dir=OUTPUT_DIR,
        save_safetensors=True,
        report_to="wandb",
        
        # 其他
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    
    logger.info("Trainer initialized - AGGRESSIVE CONFIG")
    logger.info("Config: beta=0.05, lr=2e-6, batch=8x8x4=256 (4 GPUs)")
    logger.info("WARNING: Monitor training closely for instability!")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logger.info("Training completed")

if __name__ == "__main__":
    main()
