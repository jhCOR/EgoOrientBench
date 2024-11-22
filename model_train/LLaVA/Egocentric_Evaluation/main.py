import json
import os
import sys
from transformers import TrainerCallback, TrainingArguments, EvalPrediction
import numpy as np
from typing import List, Dict
from datetime import datetime
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import argparse
import requests
import torch
import json
import re
import os
import csv
import random
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import string
from tqdm import tqdm
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

def image_parser(_setting):
    out = _setting.get("image_file").split( _setting.get("sep") )
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(_setting, _model , _tokenizer, _image_processor):
    model_name = "models--liuhaotian--llava-v1.5-7b"
    qs = _setting.get("query")
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        if _model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if _model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if _setting.get("conv_mode") is not None and conv_mode != _setting.get("conv_mode"):
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, _setting.get("conv_mode"), _setting.get("conv_mode")
            )
        )
    else:
        _setting['conv_mode'] = conv_mode

    conv = conv_templates[_setting.get("conv_mode")].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(_setting)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        _image_processor,
        _model.config
    ).to(_model.device, dtype=torch.bfloat16)
    _model.to(torch.bfloat16) 

    input_ids = (
        tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    _model.eval()
    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if _setting.get("temperature") > 0 else False,
            temperature=_setting.get("temperature"),
            top_p=_setting.get("top_p"),
            num_beams=_setting.get("num_beams"),
            max_new_tokens=_setting.get("max_new_tokens"),
            use_cache=True,
        )
    outputs = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def get_coco_name(dir_path, number):
    formatted_number = str(number).zfill(12)
    full_name_coco = f"{dir_path}/{formatted_number}.jpg"
    return full_name_coco

def get_tuning_llava():
    tuning_path = "/data/EgoOrientBench/Result_Collector/LLaVA/final_score/seed0/llava-v1.5-7b-task-lora"
    base_path = "/data/EgoOrientBench/.cache/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
    tuning_model_name = get_model_name_from_path(tuning_path)
    print(tuning_model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        tuning_path, base_path, tuning_model_name
    )
    return tokenizer, model, image_processor, context_len

def get_zeroshot_llava():
    zeroshot_path = "/data/EgoOrientBench/.cache/models--liuhaotian--llava-v1.5-7b/snapshots/4481d270cc22fd5c4d1bb5df129622006ccd9234"
    model_name = "models--liuhaotian--llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        zeroshot_path, None, model_name
    )
    return tokenizer, model, image_processor, context_len

if __name__ == '__main__':
    config = {
        'query': None,
        'conv_mode': None,
        'sep': ',',
        'temperature': 0.2,
        'top_p': None,
        'num_beams': 1,
        'max_new_tokens': 256,
    }
    
    url = os.path.expanduser(f"~/EgoOrientBench/all_data/application/coco_qa_two_obj.json")
    dir_path = os.path.expanduser(f"~/EgoOrientBench/all_data/application/val2017")

    with open(url, "r") as f:
        data_list = json.load(f)

    tuning_tokenizer, tuning_llava, image_processor, context_len = get_tuning_llava()
    zeroshot_tokenizer, zeroshot_llava, image_processor, context_len = get_zeroshot_llava()

    result = []
    prompt = "From the perspective of the camera, Look at the given photo and choose the sentence that best describes its content between the two options."
    for image in tqdm(data_list):
        image_path = get_coco_name(dir_path, image[0])

        config['query'] = prompt + "\n A." + image[1] + "\n B." + image[2]
        config['image_file'] = image_path

        tuning_result = eval_model(config, tuning_llava, tuning_tokenizer, image_processor)
        zeroshot_result = eval_model(config, zeroshot_llava, zeroshot_tokenizer, image_processor)
        print(tuning_result)
        print(zeroshot_result)

        result_per_row = {
            "image_id": image[0],
            "image_path": image_path,
            "query": config['query'],
            "tuning_result": tuning_result,
            "zeroshot_result": zeroshot_result,
            "answer":image[1],
            "result": 1 if tuning_result == zeroshot_result else 0
        }
        result.append(result_per_row)
        
    store_path = os.path.expanduser(f"~/EgoOrientBench/all_data/application/LLaVA/result_from_camera_perspective.json")
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(result, f, indent=4)


