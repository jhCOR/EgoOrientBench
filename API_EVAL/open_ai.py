from datetime import datetime
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import base64
import json
import os

from openai import OpenAI
client = None

from source.calculation import calculateIntervel, restrict_compare_Prediction_Answer, getF1Score, getF1ScorePerCategory
from source.utils import remove_punctuation
from source.logger import drawHeatMap, loggingResult
from source.ResultCollector import ResultCollector

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def openai_eval_model_with_image(setting, prompt, image_path):

    base64_image = encode_image(image_path)

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
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        temperature=setting.get("temperature", 0.2),
        max_tokens=setting.get("max_new_tokens", 150),
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content.strip()

def inference( _setting):
    result_collector = ResultCollector()

    annotation_data = None
    with open(_setting['json_path'], "r") as f:
        annotation_data = json.load(f)
    
    if _setting['mode'] == "test":
        annotation_data = annotation_data[:5]
    elif _setting['mode'] == "check":
        annotation_data = annotation_data[:100]
    else:
        print("전체 데이터 사용")
    
    print("데이터 길이: ", len(annotation_data))

    for data in tqdm(annotation_data):
        image_path = os.path.join(_setting['image_dir'], data['image'].replace("./", ""))
        prompt = data['question']

        prediction = openai_eval_model_with_image(_setting, prompt, image_path)

        if("complex" in data['type']):
            if len(prediction) > 1:
                prediction = prediction.lower().strip().replace(".", "")[1:]
            else:
                prediction = prediction.lower().strip().replace(".", "")
        else:
            prediction = prediction.lower().strip().replace(".", "")

        answer = data['label'].lower().strip().replace(".", "")
        prediction = remove_punctuation(prediction)
        answer = remove_punctuation(answer)

        result_point = restrict_compare_Prediction_Answer(prediction, answer)
        result_collector.update(data, prediction, result_point)
        result_collector.intermediate_save(_setting['dir_name'])

    result_collector.end()
    result_dict_accuracy = result_collector.getAccPerCategory()
    result_dict_f1 = result_collector.getF1ScorePerCategory()

    loggingResult(_setting['mode'], f"{_setting['dir_name']}", result_collector.prediction_list)
    drawHeatMap(f"{_setting['dir_name']}", _setting['mode'], list(result_collector.result_dict.keys()))
    return {"accuracy": result_dict_accuracy, "f1_score": result_dict_f1}

def postprocess_result(final_result_list, setting):
    accuracy_collect = {}
    for item in final_result_list['accuracy']:
        for key in item.keys():
            if(accuracy_collect.get(key) is None):
                accuracy_collect[key] = []
            accuracy_collect[key].append( item[key] )
            
    CI_collection = {}
    for key in accuracy_collect.keys():
        a_list = accuracy_collect.get(key)
        print(key, a_list)
        confi_interval = calculateIntervel(a_list)
        CI_collection[key] = confi_interval
    
    with open(f'{setting["dir_name"]}/confidential_interval.json', 'w', encoding='utf-8') as f:
        json.dump(CI_collection, f, indent=4)
    serializable_setting = {key: value for key, value in setting.items() if isinstance(value, (dict, list, str, int, float, bool, type(None)))}
    with open(f'{setting["dir_name"]}/setting.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_setting, f, indent=4)
    return CI_collection

def main(_config):
    final_result_collection = {}
    for i in range(1):
        result_dict_accuracy = inference( _config )
        print("index:", i, result_dict_accuracy)

        for key in result_dict_accuracy.keys():
            if(final_result_collection.get(key) is None):
                final_result_collection[key] = []
            final_result_collection[key].append(result_dict_accuracy.get(key))
    return final_result_collection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="test") 
    args = parser.parse_args()

    with open(os.path.expanduser("~/SECRET/secrete.json"), "r") as f:
        json_data = json.load(f)
    client = OpenAI(api_key=json_data['chatgpt'])

    now = datetime.now()
    config = {
        "mode":args.mode,
        "dir_name": os.path.expanduser(f"~/EgoOrientBench/API_EVAL/Result/open_ai_{now.hour}_{now.minute}"),
        'temperature': 0.2,
        'num_beams': 1,
        'max_new_tokens': 256,
        "json_path":os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/train_benchmark/benchmark.json"),
        "image_dir":os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset"),
        "client":client,
        "model_name":"gpt-4o-2024-08-06"
    }
    try:
        print("파일 경로 확인:", os.path.exists(f'./{config["dir_name"]}'))
        os.makedirs(f'{config["dir_name"]}', exist_ok=True)
        final_result_collection = main(config)
        postprocess_result(final_result_collection, config)
    except Exception as e:
        print("에러: ", e)

    