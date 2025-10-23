from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

dataset = load_dataset(
    "json",
    data_files={
        "train": "../../research/alignment/data/TLDR/sft/train_conversational.jsonl",
        "test": "../../research/alignment/data/TLDR/sft/test_conversational.jsonl"
    }
)

# take an example from dataset
data_examples = dataset['train'][0:2]
prompt_examples = data_examples["prompt"]
completion_examples = data_examples["completion"]

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B",
    trust_remote_code=True,
    use_fast=True
)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# examples formatted
prompts_formatted = []
for prompt in prompt_examples:
    prompt_formatted = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts_formatted.append(prompt_formatted)
logger.info(f"examples of dataset after processing: {prompts_formatted}")

# prompt + completion formatted
prompt_completions_formatted = []
for prompt, completion in zip(prompt_examples, completion_examples):
    prompt_completion = prompt + completion
    prompt_completion_formatted = tokenizer.apply_chat_template(
        prompt_completion,
        tokenize=False,
        enable_thinking=False,
    )
    prompt_completions_formatted.append(prompt_completion_formatted)
logger.info(f"examples of dataset after processing (prompt + completion): {prompt_completions_formatted}")
