#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
LOAD='MAGAer13/mplug-owl2-llama2-7b'

DATA_FILE=../../../all_data/EgocentricDataset/additional_data/imagenet_QA_7b_clean_test_mplugowl.json
VALID_DATA_FILE=../../../all_data/EgocentricDataset/additional_data/benchmark_mini.json
MOCK_DATA_FILE=../../../all_data/EgocentricDataset/additional_data/imagenet_QA_7b_clean_test_mplugowl.json

python mplug_owl2/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --visual_abstractor_lr 2e-5 \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --valid_data_path $VALID_DATA_FILE \
    --mock_data_path $MOCK_DATA_FILE \
    --image_folder ../../../all_data/EgocentricDataset/imagenet_after  \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoint/mplug-owl2-finetune-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "epoch" \
    --eval_steps 30 \
    --save_strategy "epoch" \
    --save_steps 10 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --seed 0
