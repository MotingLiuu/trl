import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, ModelConfig, DPOConfig
import logging

logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

os.environ["WANDB_PROJECT"] = "Qwen3-1.7B-TLDR"
os.environ["WANDB_RUN_NAME"] = "dpo_conservative"
os.environ["WANDB_LOG_MODEL"] = "false"

MODEL_PATH = "../model/Qwen3-1.7B-sft-config3"
OUTPUT_DIR = "../model/Qwen3-1.7B-dpo-conservative"

TRAIN_DATA = "../data/TLDR/dpo/train_conversational.jsonl"
TEST_DATA = "../data/TLDR/dpo/test_conversational.jsonl"

def main():
    logger.info("Starting DPO training - CONSERVATIVE CONFIG")
    
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
    
    # 保守配置：最稳定，适合调试问题
    dpo_config = DPOConfig(
        # 损失函数 - 更强的约束
        loss_type="sigmoid",
        beta=0.2,  # 更大的beta，更保守
        
        # 学习率 - 最小的安全值
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.2,  # 更长的warmup
        
        # 批次大小 - 小batch但大有效batch
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=32,  # effective_batch=2×32×4=256 (4 GPUs)
        
        # 训练周期
        num_train_epochs=1,
        
        # 评估和保存
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=0.5,  # 更严格的梯度裁剪
        
        # 精度
        bf16=True,
        gradient_checkpointing=True,
        
        # 日志
        logging_steps=10,
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
    
    logger.info("Trainer initialized - CONSERVATIVE CONFIG")
    logger.info("Config: beta=0.2, lr=5e-7, batch=2x32=64")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    logger.info("Training completed")

if __name__ == "__main__":
    main()
