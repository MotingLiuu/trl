from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# load dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "../../research/alignment/data/TLDR/sft/train_conversational.jsonl",
        "test": "../../research/alignment/data/TLDR/sft/test_conversational.jsonl"
    }
)
dataset = dataset.remove_columns("completion")

logger.info(f"Dataset loaded: {dataset}")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "../../research/alignment/model/Qwen3-1.7B-sft-config1/checkpoint-1824",
    trust_remote_code=True,
    use_fast=True
)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded")

# process dataset
def process_chat(example):
    prompt = example["prompt"]

    formatted = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenized = tokenizer(
        formatted,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return tokenized

dataset = dataset.map(
    process_chat,
    batched=True,
    num_proc=36,
    remove_columns=["prompt"],
)
logger.info(f"Dataset after processing: {dataset}")

# create dataloader
data_loader = DataLoader(
    dataset["train"],
    batch_size=8,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
)
logger.info("DataLoader created")
test_batch = next(iter(data_loader))
logger.info(f"Test batch keys: {test_batch.keys()}")
logger.info(f"Test batch input_ids shape: {test_batch['input_ids'].shape}")
logger.info(f"Test batch attention_mask shape: {test_batch['attention_mask'].shape}")
