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

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union

import datasets
import yaml
from datasets import DatasetDict, concatenate_datasets
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available


def _ensure_transformers_parallelism_config() -> None:
    """
    Ensure that ``transformers.training_args`` always defines the symbol `ParallelismConfig` so that Python's
    `typing.get_type_hints` can resolve annotations on `transformers.TrainingArguments` without raising a `NameError`.

    This is needed when running with ``accelerate<1.10.1``, where the module ``accelerate.parallelism_config`` did not
    exist and therefore the type alias is not imported by Transformers.

    See upstream fix PR in transformers#40818.
    """
    from typing import Any

    import transformers.training_args

    if not hasattr(transformers.training_args, "ParallelismConfig"):
        transformers.training_args.ParallelismConfig = Any


_ensure_transformers_parallelism_config()  # before creating HfArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.

    This class matches the signature of [`~datasets.load_dataset`] and the arguments are used directly in the
    `datasets.load_dataset` function. You can refer to the `datasets.load_dataset` documentation for more details.

    Parameters:
        path (`str`):
            Path or name of the dataset.
        name (`str`, *optional*):
            Defining the name of the dataset configuration.
        data_dir (`str`, *optional*):
            Defining the `data_dir` of the dataset configuration. If specified for the generic builders(csv, text etc.)
            or the Hub datasets and `data_files` is `None`, the behavior is equal to passing `os.path.join(data_dir,
            **)` as `data_files` to reference all the files in a directory.
        data_files (`str` or `Sequence` or `Mapping`, *optional*):
            Path(s) to source data file(s).
        split (`str`, *optional*, defaults to `"train"`):
            Which split of the data to load.
        columns (`list[str]`, *optional*):
            List of column names to select from the dataset. If `None`, all columns are selected.
    """

    path: str
    name: Optional[str] = None
    data_dir: Optional[str] = None
    data_files: Optional[Union[str, list[str], dict[str, str]]] = None
    split: str = "train"
    columns: Optional[list[str]] = None


@dataclass
class DatasetMixtureConfig:
    """
    Configuration class for a mixture of datasets.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        datasets (`list[DatasetConfig]`):
            List of dataset configurations to include in the mixture.
        streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the datasets. If `True`, the datasets will be loaded in streaming mode.
        test_split_size (`float`, *optional*):
            Size of the test split. Refer to the `test_size` parameter in the [`~datasets.train_test_split`] function
            for more details. If `None`, the dataset will not be split into train and test sets.

    Usage:
        When using the CLI, you can add the following section to your YAML config file:

        ```yaml
        datasets:
          - path: ...
            name: ...
            data_dir: ...
            data_files: ...
            split: ...
            columns: ...
          - path: ...
            name: ...
            data_dir: ...
            data_files: ...
            split: ...
            columns: ...
        streaming: ...
        test_split_size: ...
        ```
    """

    datasets: list[DatasetConfig] = field(
        default_factory=list,
        metadata={"help": "List of dataset configurations to include in the mixture."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the datasets. If True, the datasets will be loaded in streaming mode."},
    )
    test_split_size: Optional[float] = field(
        default=None,
        metadata={
            "help": "Size of the test split. Refer to the `test_size` parameter in the `datasets.train_test_split` "
            "function for more details. If None, the dataset will not be split into train and test sets."
        },
    )

    def __post_init__(self):
        # Convert any dataset dicts (from CLI/config parsing) into DatasetConfig objects
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                # If it's a dict, convert it to DatasetConfig
                self.datasets[idx] = DatasetConfig(**dataset)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`,, *optional*):
            Path or name of the dataset to load. If `datasets` is provided, this will be ignored.
        dataset_config (`str`, *optional*):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
            If `datasets` is provided, this will be ignored.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training. If `datasets` is provided, this will be ignored.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation. If `datasets` is provided, this will be ignored.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If `datasets` is
            provided, this will be ignored.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load. If `datasets` is provided, this will be ignored."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function. If `datasets` is provided, this will be ignored."
        },
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for training. If `datasets` is provided, this will be ignored."},
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Dataset split to use for evaluation. If `datasets` is provided, this will be ignored."},
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={
            "help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If "
            "`datasets` is provided, this will be ignored."
        },
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make logging and warning output cleaner.
    Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


# ================================================================================================
# 🎯 TrlParser - Smart Command-Line Argument Parser
# ================================================================================================
# 📚 What is a Dataclass?
#    A dataclass is a Python class that mainly holds data (like a form with fields).
#    Example:
#        @dataclass
#        class Settings:
#            name: str              # Required field
#            age: int = 25          # Optional field with default
#            email: str = "test@example.com"
#
# 🔄 How TrlParser Works:
#    1. You define your settings as dataclasses
#    2. TrlParser automatically creates --name, --age, --email arguments
#    3. User runs: python script.py --name "John" --age 30
#    4. TrlParser returns: Settings(name="John", age=30, email="test@example.com")
#
# 🎭 Two Ways to Use TrlParser:
#
#    WAY 1: Command-Line Only
#        python script.py --arg1 5 --arg2 beta
#
#    WAY 2: Config File (Recommended for complex setups)
#        python script.py --config config.yaml
#        (Config file can set default values, then CLI args override them)
#
# ================================================================================================
class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]`, *optional*):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser


    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"


    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    # ============================================================================
    # 🏗️ CONSTRUCTOR (__init__) - Initialize the Parser
    # ============================================================================
    # This method sets up the parser when you create it.
    #
    # Example usage:
    #   parser = TrlParser(dataclass_types=[MyArgs1, MyArgs2])
    #
    # What happens here:
    #   1. Normalizes dataclass_types to always be a list (even if you pass one)
    #   2. Validates that no dataclass has a "config" field (reserved name)
    #   3. Calls parent class (HfArgumentParser) to do the heavy lifting
    # ============================================================================
    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            # __dataclass_fields__ is a dict of all fields in the dataclass
            # Example: {"arg1": Field(...), "arg2": Field(...)}
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        # ========================================================================
        # STEP 3: Initialize Parent Class (HfArgumentParser)
        # ========================================================================
        # Call the parent class constructor to do the actual setup.
        # HfArgumentParser (from transformers) handles:
        #   - Reading dataclass fields
        #   - Creating argparse arguments
        #   - Type validation
        #   - Help message generation
        #
        # We inherit all that functionality and add our own features on top!
        # ========================================================================
        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self,
        args: Optional[Iterable[str]] = None,
        return_remaining_strings: bool = False,
        fail_with_unknown_args: bool = True,
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        # ========================================================================
        # STEP 1: Get Command-Line Arguments
        # ========================================================================
        # If args not provided, get them from sys.argv (what user typed)
        # sys.argv[0] is the script name, sys.argv[1:] are the arguments
        #
        # Example: python script.py --arg1 5 --config my.yaml
        # sys.argv = ["script.py", "--arg1", "5", "--config", "my.yaml"]
        # sys.argv[1:] = ["--arg1", "5", "--config", "my.yaml"]
        # ========================================================================
        args = list(args) if args is not None else sys.argv[1:]
        
        # ========================================================================
        # STEP 2: Check for Config File Flag
        # ========================================================================
        # Look for --config in the arguments. If found, load the YAML file.
        #
        # 📁 What's in a Config File?
        #    Config files are YAML files that look like:
        #
        #    ```yaml
        #    # Optional: Set environment variables
        #    env:
        #        CUDA_VISIBLE_DEVICES: "0,1"
        #        WANDB_PROJECT: "my-project"
        #
        #    # Actual configuration values
        #    learning_rate: 0.001
        #    batch_size: 32
        #    num_epochs: 10
        #    ```
        #
        # 🎯 Why Use Config Files?
        #    - Cleaner than long command lines
        #    - Easy to version control
        #    - Can share configs between experiments
        #    - Can set environment variables
        # ========================================================================
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # ====================================================================
            # STEP 2b: Apply Config Values as Defaults
            # ====================================================================
            # Take all remaining values from config (after removing "env")
            # and set them as new default values in the parser
            #
            # Example: config has {"learning_rate": 0.001, "batch_size": 32}
            # These become the new defaults, but CLI args can still override them
            #
            # Returns: List of arguments from config that weren't recognized
            # ====================================================================
            
            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # ========================================================================
        # STEP 3: Parse Final Arguments
        # ========================================================================
        # Now parse all arguments (with config defaults applied if config was used)
        # This is where the actual conversion to dataclass objects happens
        #
        # 🔄 What parse_args_into_dataclasses() does:
        #   1. Reads the modified argument defaults (from config if provided)
        #   2. Reads command-line arguments (these override defaults)
        #   3. Validates types (e.g., ensure int fields get integers)
        #   4. Creates dataclass instances with all values filled in
        #   5. Returns tuple of dataclass objects
        #
        # Example return value:
        #   (ScriptArguments(...), SFTConfig(...), ModelConfig(...))
        # ========================================================================
        
        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # ========================================================================
        # STEP 4: Handle Remaining Strings (Unrecognized Arguments)
        # ========================================================================
        # Sometimes there are arguments that don't match any dataclass field.
        # This arguments are add by some code like parser.add_argument("--extra_arg") not backed by a dataclass. May be some arguments added by subparsers.
        # This can happen when:
        #   - Typos in argument names
        #   - Using a library that adds its own args (like Accelerate)
        #   - Config file has extra fields
        #
        # We need to decide what to do with these "leftover" arguments.
        # ========================================================================
        
        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
            # Example return: (DataClass1, DataClass2, ["--unknown_arg", "value"])
            
        elif fail_with_unknown_args and config_remaining_strings:
            raise ValueError(
                f"Unknown arguments from config file: {config_remaining_strings}. Please remove them, add them to the "
                "dataclass, or set `fail_with_unknown_args=False`."
            )
        else:
            return output

    # ============================================================================
    # 🔧 HELPER METHOD - set_defaults_with_config()
    # ============================================================================
    # This method takes values from a config file and applies them as new
    # default values in the argument parser.
    #
    # 🎯 Purpose:
    #    When you load a config file, you want those values to become the new
    #    defaults, but still allow command-line arguments to override them.
    #
    # 💡 How It Works:
    #    1. Loops through all parser actions (each represents one argument)
    #    2. If an action's name matches a config key, update its default value
    #    3. Marks the argument as "not required" (since config provided a value)
    #    4. Returns list of config keys that didn't match any argument
    #
    # 📊 Example:
    #    Config file:       {"learning_rate": 0.001, "batch_size": 32, "unknown": 123}
    #    Parser args:       --learning_rate (default=0.01), --batch_size (default=16)
    #    After this method:
    #      - learning_rate default becomes 0.001
    #      - batch_size default becomes 32
    #      - Returns ["unknown"] (didn't match any parser argument)
    #
    # 🔄 Recursive for Subparsers:
    #    Some CLI tools have subcommands (like git: git commit, git push)
    #    This method handles nested parsers recursively
    #    Example: trl sft --args (sft is a subparser)
    # ============================================================================
    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        # ========================================================================
        # Inner Helper Function - apply_defaults()
        # ========================================================================
        # This is a nested function that does the actual work recursively.
        #
        # Why nested? Because we need to handle subparsers recursively, and
        # it's cleaner to define the recursive logic in a separate function.
        #
        # Args:
        #   parser: The argument parser to update
        #   kw: Dictionary of config values (e.g., {"learning_rate": 0.001})
        #
        # Returns:
        #   Set of keys that were successfully matched to arguments
        # ========================================================================
        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # ============================================================
                # CASE 1: This action is a subparser
                # ============================================================
                # Some CLI tools have subcommands:
                #   trl sft --args     (sft is a subcommand)
                #   trl dpo --args     (dpo is another subcommand)
                #
                # We need to recursively update defaults in subparsers too
                # ============================================================
                
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    # action.choices is a dict: {"sft": <subparser>, "dpo": <subparser>}
                    for subparser in action.choices.values():
                        # Recursively apply defaults to each subparser
                        used_keys.update(apply_defaults(subparser, kw))
                        
                # ============================================================
                # CASE 2: This action is a regular argument
                # ============================================================
                # action.dest is the name of the argument
                # Example: for --learning_rate, action.dest = "learning_rate"
                #
                # Check if this argument name exists in our config dictionary
                # ============================================================
                elif action.dest in kw:
                    # We found a match! Update the default value
                    # Example: --learning_rate currently defaults to 0.01
                    #          Config has learning_rate: 0.001
                    #          We update default to 0.001
                    action.default = kw[action.dest]
                    
                    # Mark as not required since config provided a value
                    # Before: --learning_rate is required (no default)
                    # After:  --learning_rate is optional (has default from config)
                    action.required = False
                    
                    # Remember that we used this key
                    used_keys.add(action.dest)
            return used_keys

        # ========================================================================
        # Apply Defaults and Find Remaining Keys
        # ========================================================================
        # Call our helper function to apply all config values as defaults
        # ========================================================================
        used_keys = apply_defaults(self, kwargs)
        
        # ========================================================================
        # Build List of Unused Config Keys
        # ========================================================================
        # Any config key that didn't match an argument is "remaining"
        # These might be typos or extra fields that don't belong
        #
        # Example:
        #   Config: {"learning_rate": 0.001, "unknown_field": 123}
        #   Parser args: [--learning_rate]
        #   used_keys: {"learning_rate"}
        #   remaining: ["--unknown_field", "123"]
        #
        # We return them in CLI format (--key value) so they can be reported
        # as unrecognized arguments
        # ========================================================================
        
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        # List comprehension explained:
        #   For each key-value pair in kwargs that wasn't used,
        #   create two items: ["--key", "value"]
        #   Example: {"unknown": 123} → ["--unknown", "123"]
        
        return remaining


# ================================================================================================
# 📚 SUMMARY: How TrlParser Works (Complete Flow)
# ================================================================================================
#
# 🎬 COMPLETE USAGE EXAMPLE:
#
# Step 1: Define Your Configuration as Dataclasses
# ────────────────────────────────────────────────
#
#   @dataclass
#   class TrainingArgs:
#       learning_rate: float = 0.01    # Default: 0.01
#       batch_size: int = 16           # Default: 16
#       num_epochs: int = 10           # Default: 10
#
# Step 2: Create Parser
# ────────────────────────────────────────────────
#
#   parser = TrlParser(dataclass_types=[TrainingArgs])
#
# Step 3a: Use Without Config File (Simple)
# ────────────────────────────────────────────────
#
#   $ python script.py --learning_rate 0.001 --batch_size 32
#
#   Result: TrainingArgs(learning_rate=0.001, batch_size=32, num_epochs=10)
#           └── learning_rate: from CLI
#           └── batch_size: from CLI  
#           └── num_epochs: dataclass default (10)
#
# Step 3b: Use With Config File (Recommended)
# ────────────────────────────────────────────────
#
#   Create config.yaml:
#   ```yaml
#   env:
#       CUDA_VISIBLE_DEVICES: "0,1"
#   learning_rate: 0.001
#   num_epochs: 20
#   ```
#
#   $ python script.py --config config.yaml --batch_size 32
#
#   Result: TrainingArgs(learning_rate=0.001, batch_size=32, num_epochs=20)
#           └── learning_rate: from config.yaml
#           └── batch_size: from CLI (overrides config)
#           └── num_epochs: from config.yaml
#           └── CUDA_VISIBLE_DEVICES env var is set!
#
# 🏆 PRIORITY ORDER (Highest to Lowest):
#   1. Command-line arguments (--arg value)        ← HIGHEST - Always wins!
#   2. Config file values (from YAML)              ← MEDIUM
#   3. Dataclass default values                    ← LOWEST - Fallback
#
# 🎯 KEY BENEFITS:
#   ✅ Auto-generates all CLI arguments from dataclasses (no manual argparse!)
#   ✅ Type checking and validation built-in
#   ✅ Config files keep your commands clean
#   ✅ CLI can override any config value
#   ✅ Can set environment variables from config
#   ✅ Supports complex setups with multiple dataclasses
#   ✅ Handles subcommands/subparsers recursively
#   ✅ Clear error messages for typos or missing values
#
# 🎓 REAL-WORLD EXAMPLE (Training a Model):
#
#   Without TrlParser (old way):
#   $ python train.py --model_name_or_path gpt2 --learning_rate 0.001 \
#     --batch_size 32 --num_epochs 10 --gradient_accumulation_steps 4 \
#     --warmup_steps 100 --logging_steps 50 --save_steps 500 \
#     --output_dir ./output --push_to_hub True --hub_token abc123 ...
#   😰 Too long! Easy to make typos!
#
#   With TrlParser (new way):
#   Create config.yaml with all settings, then:
#   $ python train.py --config config.yaml
#   😊 Clean and simple!
#
#   Need to change just one value?
#   $ python train.py --config config.yaml --learning_rate 0.0005
#   😎 Override specific values while keeping others!
#
# ================================================================================================


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"


# ================================================================================================
# 📊 get_dataset() - Load and Combine Multiple Datasets
# ================================================================================================
# 
# 🎯 PURPOSE:
#    This function loads one or more datasets and combines them into a single dataset.
#    Think of it like merging multiple Excel spreadsheets into one big spreadsheet.
#
# 💡 WHY COMBINE DATASETS?
#    - Train on diverse data sources (e.g., Wikipedia + Books + Code)
#    - Mix different conversation styles (formal + casual + technical)
#    - Combine multiple languages or domains
#    - Create richer training data for better model performance
#
# 📝 EXAMPLE USAGE:
#
#    # Single dataset
#    config = DatasetMixtureConfig(
#        datasets=[DatasetConfig(path="trl-lib/tldr")]
#    )
#    dataset = get_dataset(config)
#    # Result: One dataset with 116k examples
#
#    # Multiple datasets combined
#    config = DatasetMixtureConfig(
#        datasets=[
#            DatasetConfig(path="dataset1"),  # 10k examples
#            DatasetConfig(path="dataset2"),  # 20k examples
#        ]
#    )
#    dataset = get_dataset(config)
#    # Result: Combined dataset with 30k examples (10k + 20k)
#
# 🔄 WHAT HAPPENS:
#    1. Load each dataset from HuggingFace Hub (or local path)
#    2. Optionally filter to specific columns
#    3. Concatenate all datasets into one
#    4. Optionally split into train/test sets
#    5. Return as DatasetDict (dictionary with 'train' and optionally 'test' keys)
#
# ================================================================================================
def get_dataset(mixture_config: DatasetMixtureConfig) -> DatasetDict:
    """
    Load a mixture of datasets based on the configuration.

    Args:
        mixture_config (`DatasetMixtureConfig`):
            Script arguments containing dataset configuration.

    Returns:
        `DatasetDict`:
            Combined dataset(s) from the mixture configuration, with optional train/test split if `test_split_size` is
            set.

    Example:
    ```python
    from trl import DatasetMixtureConfig, get_dataset
    from trl.scripts.utils import DatasetConfig

    mixture_config = DatasetMixtureConfig(datasets=[DatasetConfig(path="trl-lib/tldr")])
    dataset = get_dataset(mixture_config)
    print(dataset)
    ```

    ```
    DatasetDict({
        train: Dataset({
            features: ['prompt', 'completion'],
            num_rows: 116722
        })
    })
    ```
    """
    # ============================================================================
    # STEP 1: Initialize - Prepare to Load Multiple Datasets
    # ============================================================================
    # Log how many datasets we're going to load
    # Example: "Creating dataset mixture with 3 datasets"
    # ============================================================================
    logger.info(f"Creating dataset mixture with {len(mixture_config.datasets)} datasets")
    
    # Create empty list to store loaded datasets
    # Think of this as preparing empty boxes to put datasets in
    datasets_list = []
    
    # ============================================================================
    # STEP 2: Load Each Dataset One by One
    # ============================================================================
    # Loop through each dataset configuration
    # mixture_config.datasets might be:
    #   [DatasetConfig(path="dataset1"), DatasetConfig(path="dataset2")]
    #
    # We'll load each one and add it to our list
    # ============================================================================
    for dataset_config in mixture_config.datasets:
        # ------------------------------------------------------------------------
        # STEP 2a: Log which dataset we're loading
        # ------------------------------------------------------------------------
        # Example log: "Loading dataset for mixture: trl-lib/tldr (config name: default)"
        # ------------------------------------------------------------------------
        logger.info(f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})")
        
        # ------------------------------------------------------------------------
        # STEP 2b: Actually Load the Dataset
        # ------------------------------------------------------------------------
        # 📥 What is datasets.load_dataset()?
        #    HuggingFace's function to load datasets from:
        #    - HuggingFace Hub (online): path="username/dataset-name"
        #    - Local files: path="/path/to/data.csv"
        #    - Built-in datasets: path="imdb", "squad", etc.
        #
        # 🔧 Parameters explained:
        #    path: Where to find the dataset
        #          Examples: "trl-lib/tldr", "local/data.json", "squad"
        #
        #    name: Which subset/configuration of the dataset
        #          Some datasets have multiple versions (e.g., "en", "fr", "v1", "v2")
        #          Example: path="squad", name="v2" loads SQuAD 2.0
        #
        #    data_dir: Subdirectory within the dataset
        #              Useful when dataset has multiple folders
        #
        #    data_files: Specific files to load
        #                Example: ["train.csv", "test.csv"]
        #
        #    split: Which split to load
        #           Examples: "train", "test", "validation", "train[:1000]" (first 1000)
        #
        #    streaming: If True, load data on-the-fly (saves memory for huge datasets)
        #               If False, load entire dataset into memory
        #
        # 📊 What gets returned?
        #    A Dataset object (like a pandas DataFrame but optimized for ML)
        #    Example:
        #      Dataset({
        #          features: ['prompt', 'completion'],
        #          num_rows: 10000
        #      })
        # ------------------------------------------------------------------------
        dataset = datasets.load_dataset(
            path=dataset_config.path,
            name=dataset_config.name,
            data_dir=dataset_config.data_dir,
            data_files=dataset_config.data_files,
            split=dataset_config.split,
            streaming=mixture_config.streaming,
        )
        
        # ------------------------------------------------------------------------
        # STEP 2c: Filter to Specific Columns (Optional)
        # ------------------------------------------------------------------------
        # 🎯 Why filter columns?
        #    Datasets might have many columns you don't need.
        #    This saves memory and processing time.
        #
        # Example:
        #    Original dataset columns: ['id', 'text', 'label', 'metadata', 'date']
        #    You only need: ['text', 'label']
        #    After filtering: Dataset with only ['text', 'label']
        #
        # 💡 When is this None?
        #    When dataset_config.columns is not specified, meaning "use all columns"
        # ------------------------------------------------------------------------
        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)
        
        # ------------------------------------------------------------------------
        # STEP 2d: Add to Our List
        # ------------------------------------------------------------------------
        # Append this loaded dataset to our collection
        # datasets_list now contains all loaded datasets
        # Example: [dataset1, dataset2, dataset3]
        # ------------------------------------------------------------------------
        datasets_list.append(dataset)

    # ============================================================================
    # STEP 3: Combine All Datasets (If We Loaded Any)
    # ============================================================================
    # Check if we successfully loaded at least one dataset
    # ============================================================================
    if datasets_list:
        # ------------------------------------------------------------------------
        # STEP 3a: Concatenate (Stack) Datasets Vertically
        # ------------------------------------------------------------------------
        # 📚 What is concatenate_datasets()?
        #    Combines multiple datasets by stacking rows
        #    Like stacking multiple sheets of paper
        #
        # Visual Example:
        #    Dataset 1: 1000 rows with columns [prompt, completion]
        #    Dataset 2: 2000 rows with columns [prompt, completion]
        #    
        #    After concatenate:
        #    Combined: 3000 rows with columns [prompt, completion]
        #    (First 1000 from dataset1, next 2000 from dataset2)
        #
        # ⚠️ Important: All datasets must have the same columns!
        #    Otherwise you'll get an error
        # ------------------------------------------------------------------------
        combined_dataset = concatenate_datasets(datasets_list)
        
        # ------------------------------------------------------------------------
        # STEP 3b: Log the Final Size (If Not Streaming)
        # ------------------------------------------------------------------------
        # 📊 Why check isinstance(combined_dataset, datasets.Dataset)?
        #    - Regular Dataset: Has a length (you can count rows)
        #    - IterableDataset (streaming): No length (loads on-the-fly)
        #
        # We only log size if we can actually count the rows
        # Example log: "Created dataset mixture with 50000 examples"
        # ------------------------------------------------------------------------
        if isinstance(combined_dataset, datasets.Dataset):  # IterableDataset does not have a length
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

        # ------------------------------------------------------------------------
        # STEP 3c: Optionally Split into Train/Test Sets
        # ------------------------------------------------------------------------
        # 🎯 What is train/test split?
        #    - Train set: Used to train the model
        #    - Test set: Used to evaluate the model (model never sees during training)
        #
        # 💡 Why split?
        #    To measure if your model actually learned patterns or just memorized
        #
        # 📊 How does test_split_size work?
        #    Can be:
        #    - Float (0.0 to 1.0): Percentage
        #      Example: 0.2 means "use 20% for test, 80% for train"
        #    - Integer: Number of examples
        #      Example: 1000 means "use 1000 examples for test"
        #
        # Example with 10,000 rows and test_split_size=0.2:
        #    Before: Combined dataset with 10,000 rows
        #    After:  DatasetDict({
        #              'train': 8,000 rows,
        #              'test': 2,000 rows
        #            })
        # ------------------------------------------------------------------------
        if mixture_config.test_split_size is not None:
            # We want to split the dataset
            logger.info(f"Splitting dataset into train and test sets with test size: {mixture_config.test_split_size}")
            
            # 🔧 train_test_split() does the splitting
            #    Returns a DatasetDict with 'train' and 'test' keys
            #    Shuffles data randomly before splitting (ensures random distribution)
            combined_dataset = combined_dataset.train_test_split(test_size=mixture_config.test_split_size)
            
            # Return the split dataset
            # Structure:
            #   DatasetDict({
            #       'train': Dataset(...),
            #       'test': Dataset(...)
            #   })
            return combined_dataset
        else:
            # No split requested, return all data as 'train'
            # 
            # 📦 Why wrap in DatasetDict({"train": ...})?
            #    For consistency! Training scripts expect a dictionary format:
            #      dataset['train'] → training data
            #      dataset['test'] → test data (if exists)
            #
            # This way, whether split or not, the return format is the same
            return DatasetDict({"train": combined_dataset})
    else:
        # ------------------------------------------------------------------------
        # STEP 4: Handle Error - No Datasets Loaded
        # ------------------------------------------------------------------------
        # If we reach here, datasets_list is empty
        # This means none of the datasets could be loaded
        # Raise an error to inform the user
        # ------------------------------------------------------------------------
        raise ValueError("No datasets were loaded from the mixture configuration")


# ================================================================================================
# 📚 COMPLETE EXAMPLE: How get_dataset() Works in Practice
# ================================================================================================
#
# 🎬 SCENARIO 1: Load Single Dataset
# ────────────────────────────────────────────────────────────────────────────────────────────
#
#   from trl.scripts.utils import DatasetConfig, DatasetMixtureConfig, get_dataset
#
#   config = DatasetMixtureConfig(
#       datasets=[
#           DatasetConfig(path="trl-lib/tldr")
#       ]
#   )
#   dataset = get_dataset(config)
#
#   Result:
#   DatasetDict({
#       'train': Dataset({
#           features: ['prompt', 'completion'],
#           num_rows: 116722
#       })
#   })
#
# ────────────────────────────────────────────────────────────────────────────────────────────
# 🎬 SCENARIO 2: Combine Multiple Datasets
# ────────────────────────────────────────────────────────────────────────────────────────────
#
#   config = DatasetMixtureConfig(
#       datasets=[
#           DatasetConfig(path="dataset1"),    # 10,000 examples
#           DatasetConfig(path="dataset2"),    # 15,000 examples
#           DatasetConfig(path="dataset3"),    # 5,000 examples
#       ]
#   )
#   dataset = get_dataset(config)
#
#   Result:
#   DatasetDict({
#       'train': Dataset({
#           num_rows: 30000    # 10k + 15k + 5k combined!
#       })
#   })
#
# ────────────────────────────────────────────────────────────────────────────────────────────
# 🎬 SCENARIO 3: Combine Datasets + Train/Test Split
# ────────────────────────────────────────────────────────────────────────────────────────────
#
#   config = DatasetMixtureConfig(
#       datasets=[
#           DatasetConfig(path="dataset1"),
#           DatasetConfig(path="dataset2"),
#       ],
#       test_split_size=0.2    # Use 20% for testing
#   )
#   dataset = get_dataset(config)
#
#   Result:
#   DatasetDict({
#       'train': Dataset({
#           num_rows: 20000    # 80% of combined data
#       }),
#       'test': Dataset({
#           num_rows: 5000     # 20% of combined data
#       })
#   })
#
# ────────────────────────────────────────────────────────────────────────────────────────────
# 🎬 SCENARIO 4: Load Specific Columns Only
# ────────────────────────────────────────────────────────────────────────────────────────────
#
#   config = DatasetMixtureConfig(
#       datasets=[
#           DatasetConfig(
#               path="my-dataset",
#               columns=["text", "label"]    # Only load these columns
#           )
#       ]
#   )
#   dataset = get_dataset(config)
#
#   # Original dataset might have: ['id', 'text', 'label', 'metadata', 'source']
#   # Result only has: ['text', 'label']
#   # → Saves memory and processing time!
#
# ────────────────────────────────────────────────────────────────────────────────────────────
# 🎬 SCENARIO 5: Streaming Mode (For Huge Datasets)
# ────────────────────────────────────────────────────────────────────────────────────────────
#
#   config = DatasetMixtureConfig(
#       datasets=[
#           DatasetConfig(path="huge-dataset")
#       ],
#       streaming=True    # Don't load entire dataset into memory
#   )
#   dataset = get_dataset(config)
#
#   # Dataset is loaded on-the-fly as you iterate through it
#   # → Can work with datasets larger than your RAM!
#
# ================================================================================================
#
# 🎯 KEY TAKEAWAYS:
#
#   ✅ Can load single or multiple datasets
#   ✅ Automatically combines datasets by stacking rows
#   ✅ Can filter to specific columns (saves memory)
#   ✅ Can split into train/test sets automatically
#   ✅ Supports streaming for huge datasets
#   ✅ Always returns DatasetDict format (consistent interface)
#   ✅ Provides helpful logging messages
#
# 🚨 IMPORTANT NOTES:
#
#   ⚠️ All datasets must have compatible columns for concatenation
#   ⚠️ Streaming datasets don't have length (can't count rows)
#   ⚠️ Train/test split shuffles data randomly
#
# ================================================================================================
