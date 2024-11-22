import json
import os
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import requests
import torch
import re
from PIL import Image
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from mplug_owl2.model.builder import load_pretrained_model
import numpy as np
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle

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
    
    image_file = _setting['image_file']
    conv = conv_templates["mplug_owl2"].copy()

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], _image_processor)
    image_tensor = image_tensor.to(_model.device, dtype=torch.bfloat16)
    _model.to(torch.bfloat16)

    inp = DEFAULT_IMAGE_TOKEN + _setting.get("query")
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(_model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, _tokenizer, input_ids)

    _model.eval()
    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True if _setting.get("temperature") > 0 else False,
            temperature=_setting.get("temperature"),
            max_new_tokens=_setting.get("max_new_tokens"),
            use_cache=False,
            stopping_criteria=[stopping_criteria])

    outputs = _tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

def get_coco_name(dir_path, number):
    formatted_number = str(number).zfill(12)
    full_name_coco = f"{dir_path}/{formatted_number}.jpg"
    return full_name_coco

def get_tuning_mplug():
    tuning_path = "/data/EgoOrientBench/Result_Collector/mPLUG_Owl/final_score/seed0/mplug-owl2-finetune-lora"
    base_path = "/data/EgoOrientBench/.cache/models--MAGAer13--mplug-owl2-llama2-7b/snapshots/200342bbdd0eef019b02b4d7c9b17df235bba4ad"
    tuning_model_name = get_model_name_from_path(tuning_path)
    print(tuning_model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(tuning_path, base_path, tuning_model_name, load_8bit=False, load_4bit=False, device="cuda")

    return tokenizer, model, image_processor, context_len

def get_zeroshot_mplug():
    model_name = "models--MAGAer13--mplug-owl2-llama2-7b"
    model_path = "/data/EgoOrientBench/.cache/models--MAGAer13--mplug-owl2-llama2-7b/snapshots/200342bbdd0eef019b02b4d7c9b17df235bba4ad"
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

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
    
    url = os.path.expanduser("~/EgoOrientBench/application/coco_qa_two_obj.json")
    dir_path = os.path.expanduser("~/EgoOrientBench/application/val2017")

    with open(url, "r") as f:
        data_list = json.load(f)

    tuning_tokenizer, tuning_llava, image_processor, context_len = get_tuning_mplug()
    zeroshot_tokenizer, zeroshot_llava, image_processor, context_len = get_zeroshot_mplug()

    result = []
    prompt = "From the perspective of the camera, look at the given photo and choose the sentence that best describes its content between the two options."
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
    
    store_path = os.path.expanduser("~/EgoOrientBench/application/mplugowl2/result_from_camera_perspective_mplugowl.json")
    print("Save at: ", store_path)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(result, f, indent=4)


