python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/llava-v1.5-7b-task-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --save_model_path ./stage1_checkpoint/freeze_encoder \
