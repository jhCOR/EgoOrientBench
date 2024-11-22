import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_owl2.conversation import conv_templates, SeparatorStyle
import numpy as np
from datetime import datetime
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

            # 파일에 쓰기
            f.write(line)

def list_files_in_directory(directory_path):
    # directory_path의 파일 및 폴더 리스트 가져오기
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
        
def eval_model(_setting, data_object, _model , _tokenizer, _image_processor):
    
    conv = conv_templates["mplug_owl2"].copy()

    image = data_object["image"].convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], _image_processor)
    image_tensor = image_tensor.to(_model.device, dtype=torch.float16)
    
    if _setting['train_process']:
        _model = _model.to(dtype=torch.bfloat16)
        image_tensor = image_tensor.to(torch.bfloat16)
    
    inp = DEFAULT_IMAGE_TOKEN +  data_object.get("question")
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

def inference(model, tokenizer, image_processor, setting=None, train=None):
    
    if train is True:
        setting['train_process'] = True
    ds = load_dataset("lmms-lab/MME", split='test')

    complete_result_dict = {}
    for data_object in tqdm(ds):
        prediction = eval_model(setting, data_object, model, tokenizer, image_processor)

        data_object['prediction'] = prediction.replace("</s>", "")

        if(complete_result_dict.get(data_object['category']) is None):
            complete_result_dict[data_object['category']] = []
        complete_result_dict[data_object['category']].append(data_object)

    return complete_result_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="zeroshot") 
    args = parser.parse_args()

    print(args.mode)

    setting = {
            'query': None,
            'conv_mode': None, 
            'image_file': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256,
            'train_process': True,
        }
    
    Tuning = None
    if(args.mode == "tuning"):
        Tuning = True
        middle_name = "Tuning"
        model_path = "./checkpoints/mplug-owl2-finetune-lora"
    elif(args.mode == "zeroshot"):
        Tuning = False
        middle_name = "ZeroShot"
        model_path = 'MAGAer13/mplug-owl2-llama2-7b'

    final_result_collection = {}
    seed_list = [0]
    print("seed_list: ", seed_list)

    for seed in seed_list:
        restrict_seed(seed)

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")
        
        final_result = inference(model, tokenizer, image_processor, setting)

        now = datetime.now()
        subfix = model_path.split("/")[-2]
        eval_save_root_path = os.path.expanduser(f"~/EgoOrientBench/model_eval/MME/MPLUG_OWL_EVAL/Zeroshot/seed{seed}_{now.hour}_{now.minute}/")
        print("eval_save_root_path: ", eval_save_root_path)
        save_to_file(final_result, eval_save_root_path)