accelerate launch --config_file deepspeed_zero0.yaml \
    dpo.py \
    --config Qwen2-0.5B-Instruct.yaml