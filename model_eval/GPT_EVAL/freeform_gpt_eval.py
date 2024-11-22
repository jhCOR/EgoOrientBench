from datetime import datetime
from tqdm import tqdm
import json
import os
import argparse
from openai import OpenAI
client = None

def openai_eval_model_with_image(setting, prompt):
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
    question = '''You are given an answer and a prediction representing an object’s orientation out of 8 possible directions. 
    Respond with 'yes' if the answer and prediction match, or 'no' if they do not. \n

        [Example]
        If the answer is 'front right' and the prediction is 'facing right while facing the camera,' respond with 'yes.'
        If the answer is 'front right' and the prediction is 'facing the camera,' respond with 'no,' 
        because 'front' and 'front right' differ in orientation."'''
    
    input_sentence = f"Answer: {answer} Prediction:{prediction}"
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
    for item in tqdm(json_data):
        if item['type'] == "freeform":
            
            prompt = make_template(item['original_label'], item['prediction'])
            eval_response = openai_eval_model_with_image(_setting, prompt)
            item['result'] = check_right(eval_response)
            
            total_count += 1
            right_count += item['result']

            print(eval_response)
            with open(f"{_setting['intermediate_dir']}/intermediate.json", "w") as f:

                json.dump(json_data, f, indent=4)

    print(round(right_count/total_count, 1))

    return json_data

def get_model_name(path):
    if "LLaVA" in path:
        return "LLaVA"
    elif "mPLUG-Owl" in path:
        return "mPLUG_Owl"
    elif "MiniGPT4" in path:
        return "MiniGPT4"
    elif "chatgpt" in path:
        return "chatgpt"
    elif "claude" in path:
        return "claude"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test") 
    args = parser.parse_args()

    print(args.mode)

    with open("path/to/secret", "r") as f:
        json_data = json.load(f)
    client = OpenAI(api_key=json_data['chatgpt'])

    now = datetime.now()
    config = {
        "mode":args.mode,
        "dir_name": os.path.expanduser(f"~/EgoOrientBench/model_eval/GPT_EVAL/to_data"),
        "intermediate_dir": os.path.expanduser(f"~/EgoOrientBench/model_eval/GPT_EVAL/intermediate/{now.strftime('%d-%H-%M-%S')}"),
        'temperature': 0.2,
        'num_beams': 1,
        "client":client,
        "model_name":"gpt-4o-2024-08-06",
    }

    url_list = [
        "./from_datasample.json"
    ]
    print(url_list)
    print("=====================================\n")

    for url in url_list:
        print(url)
        model_name = get_model_name(url)

        with open(url, "r") as f:
            json_data = json.load(f)

            try_num = url.split("/")[-2]
            dir_path = f"{config['dir_name']}/{model_name}/{try_num}"

            config['intermediate_dir'] = f"{dir_path}_{model_name}_{try_num}"
            os.makedirs(config['intermediate_dir'], exist_ok=True)
            
            result_data = gpt_eval_process(config, json_data, args.mode)
            print("--완료--")
            os.makedirs(dir_path, exist_ok=True)

            with open(f"{dir_path}/{url.split('/')[-1]}", "w") as f:
                json.dump(result_data, f, indent=4)


    