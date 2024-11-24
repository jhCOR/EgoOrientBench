from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path
)
from datetime import datetime
from tqdm import tqdm
import argparse
import torch
import re
import os
from datasets import load_dataset
import numpy as np
import random

def restrict_seed(seed):
    print("시드 고정: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def write_dict_to_txt(dict_list, file_path):
    with open(file_path, 'w') as f:
        for entry in dict_list:

            image_file = entry['question_id'].split("/")[-1]
            question = entry['question']
            answer = entry['answer']
            prediction = entry['prediction']
            line = f"{image_file}\t{question}\t{answer}\t{prediction}\n"

            f.write(line)

def list_files_in_directory(directory_path):
    file_list = os.listdir(directory_path)
    return file_list

def save_to_file(data_dict, root_path):

    if(os.path.exists(root_path) is False):
        os.makedirs(root_path, exist_ok=True)

    path = os.path.expanduser("~/EgoOrientBench/model_eval/MME/sample")
    file_list = list_files_in_directory(path)
    for file in file_list:
        print(root_path+file)
        write_dict_to_txt(data_dict[file.split(".")[0]], root_path+file)

def eval_model(_setting, data_object, _model , _tokenizer, _image_processor, model_name):
    qs = data_object.get("question")
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

    images = [data_object["image"].convert("RGB")]
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        _image_processor,
        _model.config
    ).to(_model.device, dtype=torch.float16)

    if _setting['train_process']:
        _model = _model.to(dtype=torch.bfloat16)
        images_tensor = images_tensor.to(torch.bfloat16)
    
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

def inference(setting, model, tokenizer, image_processor, train=None):
    
    if train is True:
        setting['train_process'] = True
    ds = load_dataset("lmms-lab/MME", split='test')

    complete_result_dict = {}
    for data_object in tqdm(ds):
        prediction = eval_model(setting, data_object, model, tokenizer, image_processor, tokenizer.name_or_path)

        data_object['prediction'] = prediction

        if(complete_result_dict.get(data_object['category']) is None):
            complete_result_dict[data_object['category']] = []
        complete_result_dict[data_object['category']].append(data_object)

    return complete_result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="zeroshot") 
    args = parser.parse_args()

    setting = {
            'query': None,
            'conv_mode': None, 
            'image_file': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256,
            "train_process": True
        }
    
    print(args.mode)

    Tuning = None
    if(args.mode == "zeroshot"):
        Tuning = False
        middle_name = "ZeroShot"
        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        
    final_result_collection = {}

    seed_list = [0]
    print("seed_list: ", seed_list)
    for seed in seed_list:

        restrict_seed(seed)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )

        now = datetime.now()
        eval_save_root_path = f"../../model_eval/MME/LLaVA_EVAL/Zeroshot/seed{seed}_{now.hour}_{now.minute}/"

        final_result = inference(setting, model, tokenizer, image_processor, train=True)
        save_to_file(final_result, eval_save_root_path)