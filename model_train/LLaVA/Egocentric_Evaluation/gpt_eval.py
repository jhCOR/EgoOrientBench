from datetime import datetime
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
import string
from scipy.stats import t
import argparse
import copy
from openai import OpenAI
client = None
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openai_eval_model_with_image(setting, prompt):
    # ChatCompletion API 호출
    client = setting.get("client")
    response = client.chat.completions.create(
        model=setting['model_name'],
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt,
                }
            ],
            }
        ],
        temperature=setting.get("temperature", 0.2),
        max_tokens=setting.get("max_new_tokens", 50),
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content.strip()

def make_template(answer, prediction):
    question = "Check if the given prediction matches the ground truth. Respond with 'yes' only if they match, and 'no' otherwise. \n"
    
    input_sentence = f"Letter Answer: A. Sentence Answer: {answer} Prediction:{prediction}"
    prompt = question + "\n" + input_sentence
    return prompt

def check_right(response):
    if "yes" in response.lower() and "no" not in response.lower():
        return 1
    else:
        return 0

def gpt_eval_process(_setting, json_data, mode="real"):
    if mode == "test":
        json_data = json_data[:1000]

    print(len(json_data))

    total_count = 0
    right_count = 0
    zeroshot_right_count = 0
    for item in tqdm(json_data):
        query_tuning= make_template(item['answer'], item['tuning_result'])
        eval_response_tuning = openai_eval_model_with_image(_setting, query_tuning)
        item['tuning_llave_correct'] = check_right(eval_response_tuning)

        query_zeroshot = make_template(item['answer'], item['zeroshot_result'])
        eval_response_zeroshot = openai_eval_model_with_image(_setting, query_zeroshot)
        item['zeroshot_llave_correct'] = check_right(eval_response_zeroshot)
        
        total_count += 1
        right_count += item['tuning_llave_correct']
        zeroshot_right_count += item['zeroshot_llave_correct']
        
        with open(f"{_setting['intermediate_dir']}/llava_intermediate.json", "w") as f:
            json.dump(json_data, f, indent=4)

    print("Tuning:", round(right_count/total_count, 3))
    print("Zeroshot:", round(zeroshot_right_count/total_count, 3))

    return json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test") 
    args = parser.parse_args()

    print(args.mode)

    with open(os.path.expanduser(f"~/EgoOrientBench/all_data/application/SECRET/secrete.json"), "r") as f:
        json_data = json.load(f)
    client = OpenAI(api_key=json_data['chatgpt'])

    now = datetime.now()
    config = {
        "mode":args.mode,
        "dir_name": os.path.expanduser(f"~/EgoOrientBench/all_data/application/Results"),
        "intermediate_dir": os.path.expanduser(f"~/EgoOrientBench/all_data/application/Results/intermediate/{now.strftime('%d-%H-%M-%S')}"),
        'temperature': 0.2,
        'num_beams': 1,
        "client":client,
        "model_name":"gpt-4o-2024-08-06",
    }

    url_list = [
        os.path.expanduser(f"~/EgoOrientBench/all_data/application/LLaVA/result_from_camera_perspective.json")
    ]
    print(url_list)
    print("=====================================\n")

    for url in url_list:
        print(url)
        model_name = "application_phase_llava"

        with open(url, "r") as f:
            json_data = json.load(f)

            try_num = url.split("/")[-2]
            dir_path = f"{config['dir_name']}/{model_name}/{try_num}"

            config['intermediate_dir'] = f"eval_llava_{config['intermediate_dir']}_{model_name}_{try_num}"
            os.makedirs(config['intermediate_dir'], exist_ok=True)
            
            result_data = gpt_eval_process(config, json_data, args.mode)
            print("--완료--")
            os.makedirs(dir_path, exist_ok=True)

            print(f"{dir_path}/llava_evaluated_{url.split('/')[-1]}")
            with open(f"{dir_path}/llava_evaluated_{url.split('/')[-1]}", "w") as f:
                json.dump(result_data, f, indent=4)


    