#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

python evaluate.py --model_type api --model_name chatgpt --key_path ../../SECRET/secrete.json
#python evaluate.py --model_type opensource --model_name llava --model_path "liuhaotian/llava-v1.5-7b"
#python evaluate.py --model_type opensource --model_name internvl --checkpoint "/data/DAL_storage/pretrained/InternVL2-4B"
#python evaluate.py --model_type opensource --model_name mplugowl --model_path 'MAGAer13/mplug-owl2-llama2-7b'