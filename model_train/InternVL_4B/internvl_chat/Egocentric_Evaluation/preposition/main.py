import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../../', 'internvl_chat')
sys.path.append(model_train_dir)

import json
import torch
from PIL import Image
from tqdm import tqdm
from argparse import Namespace

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

def load_image(image_file,args, input_size=224):
    try:
        image = Image.open(image_file).convert('RGB')
    except Exception as e:
        print(f"Failed to load image {image_file}: {e}")
        return None  # Optionally, return None or an empty tensor if needed
    transform = build_transform(is_train=False, input_size=input_size)
    if args.dynamic:
        images = dynamic_preprocess(image, image_size=input_size,
                                    use_thumbnail=use_thumbnail,
                                    max_num=args.max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def eval_model(_setting, _model , _tokenizer, args):
    image_size = _model.config.force_image_size or _model.config.vision_config.image_size
    question = "<image>\n"+_setting.get("query")
    
    img_path = _setting.get("image_file")
    if not os.path.exists(img_path):
        print(f"-----> Warning: Image file '{img_path}' does not exist. Skipping this entry.")
        return ""  # 또는 continue로 루프가 있다면 넘어감

    loaded_image = load_image(img_path, args,image_size)
    if loaded_image is not None:
        pixel_values = loaded_image.cuda().to(torch.bfloat16)
    else:
        return ""

    generation_config = dict(
        num_beams=args.num_beams,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2
    )
    response = _model.chat(
        tokenizer=_tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        verbose=True
    )
    return response

def get_coco_name(dir_path, number):
    formatted_number = str(number).zfill(12)
    full_name_coco = f"{dir_path}/{formatted_number}.jpg"
    return full_name_coco

def get_tuning_internVL():

    args = Namespace(
        checkpoint="/data/pretrained/InternVL2-4B",
        root='./Your_Results', 
        num_beams=config.get("num_beams"), 
        top_k=50, 
        top_p=config.get("top_p"), 
        sample=False, 
        dynamic=False, 
        max_num=6, 
        load_in_8bit=False, 
        load_in_4bit=False, 
        auto=False)
    model, tokenizer = load_model_and_tokenizer(args)

    return tokenizer, model, args

def get_zeroshot_internVL():

    args = Namespace(
        checkpoint="/data/pretrained/InternVL2-4B",
        root='./Your_Results', 
        num_beams=config.get("num_beams"), 
        top_k=50, 
        top_p=config.get("top_p"), 
        sample=False, 
        dynamic=False, 
        max_num=6, 
        load_in_8bit=False, 
        load_in_4bit=False, 
        auto=False)
    model, tokenizer = load_model_and_tokenizer(args)

    return tokenizer, model, args

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

    url = os.path.expanduser(f"~/EgoOrientBench/application/coco_qa_two_obj.json")
    dir_path = os.path.expanduser(f"~/EgoOrientBench/application/val2017")

    with open(url, "r") as f:
        data_list = json.load(f)

    tuning_tokenizer, tuning_llava, args = get_tuning_internVL()
    zeroshot_tokenizer, zeroshot_llava, args = get_zeroshot_internVL()

    result = []
    prompt = "From the perspective of the camera, look at the given photo and choose the sentence that best describes its content between the two options."
    for image in tqdm(data_list):
        image_path = get_coco_name(dir_path, image[0])

        config['query'] = prompt + "\n A." + image[1] + "\n B." + image[2]
        config['image_file'] = image_path

        tuning_result = eval_model(config, tuning_llava, tuning_tokenizer, args)
        zeroshot_result = eval_model(config, zeroshot_llava, zeroshot_tokenizer, args)

        print(tuning_result)
        print(zeroshot_result)

        result_per_row = {
            "image_id": image[0],
            "image_path": image_path,
            "query": config['query'],
            "tuning_result": tuning_result,
            "zeroshot_result": zeroshot_result,
            "answer":image[1],
            "letter_answer":"A",
            "result": 1 if tuning_result == zeroshot_result else 0
        }
        result.append(result_per_row)
    
    store_path = "./Results/result_from_camera_perspective_internVL.json"
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "w") as f:
        json.dump(result, f, indent=4)


