from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3ForCausalLM
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
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
def process_chat(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]

    prompt_completions = [prompt + completion for prompt, completion in zip(prompts, completions)]
    prompt_completions_formatted = []
    for prompt_completion in prompt_completions:
        prompt_completion_formatted = tokenizer.apply_chat_template(
            prompt_completion,
            tokenize=False,
            enable_thinking=False,
        )
        prompt_completions_formatted.append(prompt_completion_formatted)

    # need to return a dict, column name : value
    return {"prompt_completions_formatted": prompt_completions_formatted}

def tokenize_fn(examples):
    texts = examples["prompt_completions_formatted"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    return tokenized

'''
data_example = dataset['train'][0]
logger.info(f"data example: {data_example}")
logger.info(f"Processed example: {process_chat(data_example)}")
'''
logger.info(f"examples of dataset before processing: {dataset['train'][:2]}")

dataset = dataset.map(
    process_chat,
    batched=True,
    num_proc=36,
    remove_columns=["completion", "prompt"],
)
#logger.info(f"Dataset after formatting: {dataset}")
#logger.info(f"Processed formatting: {dataset['train'][0:2]}")

#tokenized_examples = tokenizer(
#    dataset['train']['prompt_completions_formatted'][:2],
#    truncation=True,
#    max_length=tokenizer.model_max_length,
#)

#logger.info(f"\nTokenized example: {tokenized_examples}")

#padded_examples = tokenizer.pad(
#    tokenized_examples,
#    padding=True,
#    return_tensors="pt"
#)
#logger.info(f"\nPadded example: {padded_examples}")

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    num_proc=36,
    remove_columns=["prompt_completions_formatted"],
)
logger.info(f"Dataset after tokenization: {dataset}")
logger.info(f"Tokenized example: {dataset['train'][0:2]}")

# create dataloader
data_loader = DataLoader(
    dataset["train"],
    batch_size=2,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
)
logger.info(f"DataLoader created with batch size 2")

batch = next(iter(data_loader))
logger.info(f"example of a batch from dataloader: {batch}")

# load model
model = AutoModelForCausalLM.from_pretrained(
    "../../research/alignment/model/Qwen3-1.7B-dpo-config1/checkpoint-1451",
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()
logger.info("Model loaded")

# test generation
with torch.no_grad():
    outputs = model.generate(
        input_ids=batch["input_ids"].to(model.device),
        attention_mask=batch["attention_mask"].to(model.device),
        max_new_tokens=1024,
        temperature=0.1,
    )
logger.info(f"Generation outputs shape: {outputs.shape}")
generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, text in enumerate(generated_texts):
    logger.info(f"Generated text {i}: {text}\n\n")