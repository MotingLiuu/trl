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
        "test": "data/test_conversational.jsonl",
    }
)
logger.info("Dataset loaded successfully.")
logger.info(f"Sample data: {dataset['test'][0:2]}")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
logger.info("Tokenizer loaded successfully.")

# process the dataset
dataset = dataset.remove_columns(["chosen", "rejected"])
logger.debug(f"Sample data after removing columns: {dataset['test'][0:2]}")

def prompt_format(examples):
    prompt = examples["prompt"]
    
    prompt_formatted = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {"prompt_formatted": prompt_formatted}

dataset = dataset.map(prompt_format, batched=True, remove_columns=["prompt"])
logger.debug(f"Sample data after prompt formatting: {dataset['test'][0:2]}")

def tokenize_fn(examples):
    texts = examples["prompt_formatted"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return tokenized

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["prompt_formatted"])
logger.debug(f"Sample data after tokenization: {dataset['test'][0:2]}")

# create dataloader
data_loader = DataLoader(
    dataset["test"],
    batch_size=8,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
)
logger.info("DataLoader created successfully.")
logger.info(f"Sample batch from DataLoader: {next(iter(data_loader))}")

# generate responses
# load model
model = AutoModelForCausalLM.from_pretrained(
    "../model/Qwen3-1.7B-sft-config3/checkpoint-1824",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda",
) 
with torch.no_grad():
    for batch in data_loader:
        outputs = model.generate(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
        )
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for text in generated_text:
            print("----------------------------------------------------")
            print(f"Generated text: {text}")
            print("----------------------------------------------------")
        break  # only generate for one batch for demonstration