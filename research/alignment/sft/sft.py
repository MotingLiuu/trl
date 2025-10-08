# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

# ================================================================================================
# 📖 DOCSTRING - USAGE EXAMPLES
# ================================================================================================
# This docstring provides two example commands for using this script.
# ================================================================================================
"""
SFT (Supervised Fine-Tuning) Training Script

This script fine-tunes language models on your custom datasets. It supports two training modes:

1️⃣ FULL TRAINING - Updates ALL model parameters (requires more GPU memory)
   Use this when: You have sufficient GPU memory and want maximum model adaptation

2️⃣ LoRA TRAINING - Only updates small adapter layers (memory-efficient)
   Use this when: GPU memory is limited or you want faster training


# ============================================================================
# EXAMPLE 1: Full Training (All Parameters)
# ============================================================================
# What each argument does:
#   --model_name_or_path: Which pre-trained model to start from
#                         (like choosing a base cake recipe to modify)
#   --dataset_name: Which dataset to train on (your training examples)
#   --learning_rate: How big steps to take when updating the model
#                    (smaller = slower but more stable, larger = faster but risky)
#   --num_train_epochs: How many times to go through entire dataset
#   --packing: Efficiency trick - combines multiple short examples into one
#              (like packing multiple items in one box to save space)
#   --per_device_train_batch_size: How many examples to process at once per GPU
#   --gradient_accumulation_steps: Process N batches before updating weights
#                                  (simulates larger batch size without using more memory)
#   --gradient_checkpointing: Memory-saving technique (slower but uses less RAM)
#   --eos_token: "End of Sequence" marker (tells model where text ends)
#   --eval_strategy: When to run validation (every N steps or epochs)
#   --eval_steps: Validate every 100 training steps
#   --output_dir: Where to save the trained model
#   --push_to_hub: Upload trained model to HuggingFace Hub (like GitHub for models)
# ============================================================================
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```

# ============================================================================
# EXAMPLE 2: LoRA Training (Parameter-Efficient)
# ============================================================================
# Additional LoRA-specific arguments:
#   --use_peft: Enable Parameter-Efficient Fine-Tuning (LoRA mode)
#   --lora_r: LoRA rank (size of adapter - higher = more parameters but more powerful)
#             Think of it like resolution: 32 is higher quality than 8
#   --lora_alpha: LoRA scaling factor (controls how much LoRA affects the model)
#   --learning_rate: Can be 10x higher than full training (2.0e-4 vs 2.0e-5)
#                    because we're only updating small adapters
#
# 💡 Why LoRA is great:
#    - Uses 90% less GPU memory
#    - Trains 2-3x faster
#    - Final model size is smaller (just the adapter weights)
#    - Can easily swap different adapters for different tasks
# ============================================================================
```
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT \
    --push_to_hub
```
"""

import argparse
import os
from typing import Optional

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

# ============================================================================
# TRL Library - Tools for Training Language Models
# ============================================================================
# TRL (Transformer Reinforcement Learning) provides high-level training utilities
#
# DatasetMixtureConfig: Configuration for mixing multiple datasets
#                       Example: 70% dataset A + 30% dataset B
#
# ModelConfig: Configuration for model settings
#              (LoRA settings, quantization, attention implementation, etc.)
#
# ScriptArguments: Configuration for script-specific arguments
#                  (dataset name, train/test split, etc.)
#
# SFTConfig: Configuration for Supervised Fine-Tuning
#            (learning rate, batch size, number of epochs, etc.)
#
# SFTTrainer: The main trainer class that handles the entire training process
#
# TrlParser: Smart argument parser that can handle multiple configuration types
#
# get_dataset: Helper function to load datasets (supports mixing multiple datasets)
#
# get_kbit_device_map: Creates device map for quantized models
#                      (tells which parts of model go on which GPU/CPU)
#
# get_peft_config: Creates PEFT (LoRA) configuration from model arguments
#
# get_quantization_config: Creates quantization configuration (4-bit, 8-bit)
#                          (quantization = using smaller numbers to save memory)
# ============================================================================
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# ================================================================================================
# 🚀 MAIN TRAINING FUNCTION
# ================================================================================================
# This is the heart of the script - it orchestrates the entire training process.
#
# The function receives 4 configuration objects (think of them as 4 different setting panels):
#   1. script_args: General script settings (dataset name, splits, etc.)
#   2. training_args: Training hyperparameters (learning rate, batch size, epochs, etc.)
#   3. model_args: Model-specific settings (which model, LoRA config, quantization, etc.)
#   4. dataset_args: Dataset configuration (for mixing multiple datasets)
#
# The training flow has 6 main steps:
#   STEP 1: Setup model loading configuration
#   STEP 2: Load the model (detect if it's VLM or text-only)
#   STEP 3: Load the tokenizer
#   STEP 4: Load and prepare the dataset
#   STEP 5: Create and configure the trainer
#   STEP 6: Train, save, and optionally upload to HuggingFace Hub
# ================================================================================================
def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    model_kwargs = dict(
        # revision: Specific version/commit of the model (like a Git commit hash)
        #           None = use latest version
        #           "main" = use main branch
        #           "abc123" = use specific commit
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        
        # device_map: Tells PyTorch where to put each part of the model
        #             "auto" = automatically split across available GPUs/CPU
        #             This is crucial for large models that don't fit on one GPU
        model_kwargs["device_map"] = get_kbit_device_map()
        
        # Add the quantization configuration
        model_kwargs["quantization_config"] = quantization_config

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    valid_image_text_architectures = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

    if config.architectures and any(arch in valid_image_text_architectures for arch in config.architectures):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name,  # Dataset identifier (like "username/dataset-name")
            name=script_args.dataset_config,  # Optional: subset name (some datasets have multiple configs)
            streaming=script_args.dataset_streaming  # Streaming: don't download everything, stream as needed
                                                     # Useful for HUGE datasets that don't fit in memory
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the SFT trainer
    trainer = SFTTrainer(
        # The model to train
        model=model,
        # Training configuration (learning rate, batch size, epochs, etc.)
        args=training_args,
        
        # Training data
        # dataset[...] extracts a specific split (e.g., dataset["train"])
        # script_args.dataset_train_split typically = "train"
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


# ================================================================================================
# 🔧 ARGUMENT PARSER SETUP FUNCTION
# ================================================================================================
# This function creates a command-line argument parser that can handle multiple
# configuration dataclasses simultaneously.
#
# 🎯 Why multiple dataclasses?
#    Different aspects of training need different settings:
#    - Script settings (dataset name, splits)
#    - Training settings (learning rate, batch size)
#    - Model settings (LoRA config, quantization)
#    - Dataset settings (mixing ratios)
#
#    Separating them keeps code organized and reusable.
#
# 🔀 Two modes this parser supports:
#    1. Standalone mode: Used when running script directly
#       Example: python sft.py --model_name_or_path Qwen/Qwen2-0.5B
#
#    2. Subcommand mode: Used when script is part of larger CLI tool
#       Example: trl sft --model_name_or_path Qwen/Qwen2-0.5B
#                ^^^     ^subcommand
#                CLI tool
# ================================================================================================
def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    # ============================================================================
    # Define the 4 Configuration Dataclasses
    # ============================================================================
    # These dataclasses define all possible command-line arguments
    # TrlParser will automatically generate CLI arguments from their fields
    #
    # Example: If ScriptArguments has a field `dataset_name: str`,
    #          TrlParser creates a --dataset_name argument automatically
    # ============================================================================
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    
    # ============================================================================
    # STEP 2: Parse Command-Line Arguments
    # ============================================================================
    # parse_args_and_config() is a smart function that:
    #   1. Reads command-line arguments (--model_name_or_path, --learning_rate, etc.)
    #   2. Can also read from config files (--config config.json)
    #   3. Validates all arguments (checks types, required fields, etc.)
    #   4. Splits arguments into their respective dataclass objects
    #
    # return_remaining_strings=True means:
    #   "If there are unknown arguments, don't crash - just ignore them"
    #   This is useful because Accelerate (for distributed training) may add
    #   its own arguments, and we don't want to interfere with those.
    #
    # Returns 5 values:
    #   - script_args: ScriptArguments object (dataset settings)
    #   - training_args: SFTConfig object (training hyperparameters)
    #   - model_args: ModelConfig object (model settings)
    #   - dataset_args: DatasetMixtureConfig object (dataset mixing settings)
    #   - _: Remaining unknown arguments (we ignore these)
    # ============================================================================
    
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    
    # ============================================================================
    # STEP 3: Run the Main Training Function
    # ============================================================================
    # Now that we have all configurations parsed, pass them to main() to:
    #   1. Load the model and tokenizer
    #   2. Load the dataset
    #   3. Create the trainer
    #   4. Train the model
    #   5. Save and optionally upload to HuggingFace Hub
    #
    # 🎯 COMPLETE EXECUTION FLOW:
    #
    #   Command Line Input
    #         ↓
    #   parse_args_and_config()  →  4 config objects
    #         ↓
    #   main(...)
    #         ↓
    #   ┌─────────────────────────────────────┐
    #   │ 1. Setup model loading config       │
    #   │ 2. Load model (VLM or text-only)    │
    #   │ 3. Load tokenizer                   │
    #   │ 4. Load dataset                     │
    #   │ 5. Create SFTTrainer                │
    #   │ 6. Train model                      │
    #   │ 7. Save model                       │
    #   │ 8. (Optional) Push to Hub           │
    #   └─────────────────────────────────────┘
    #         ↓
    #   ✅ Trained Model Ready!
    #
    # ============================================================================
    main(script_args, training_args, model_args, dataset_args)
