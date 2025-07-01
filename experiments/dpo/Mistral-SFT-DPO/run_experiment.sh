export WANDB_MODE="offline"
accelerate launch --config_file deepspeed_zero3.yaml \
    dpo.py \
    --config Mistral-7b-sft-full.yaml 