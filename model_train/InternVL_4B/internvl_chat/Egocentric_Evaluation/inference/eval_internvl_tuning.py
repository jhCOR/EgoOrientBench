import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../../', 'internvl_chat')
sys.path.append(model_train_dir)

from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess

import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image
from argparse import Namespace
import torch
import json
import os
import random
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from datetime import datetime

def restrict_seed(seed):
    print("시드 고정: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_punctuation(text):
    # string.punctuation에는 모든 문장부호가 포함되어 있습니다.
    return text.translate(str.maketrans('', '', string.punctuation))

def restrict_compare_Prediction_Answer(prediction, answer):
    result = (answer == prediction)

    if(result):
        point = 1
    else:
        point = 0
    return point

def getF1Score(category_list, result_list):
    f1_result = {}
    for category in category_list:
        pred_list = [item["prediction"].lower().strip() for item in result_list if item['type'] == category ]
        ans_list  = [item["answer"].lower().strip().replace(".", "") for item in result_list if item['type'] == category ]
        f1_result[category] = getF1ScorePerCategory(pred_list, ans_list)
    return f1_result

def getF1ScorePerCategory(pred_list, ans_list):
    return f1_score(ans_list, pred_list, average='macro')

def logging(path, epoch, accuracy):
    accuracy['epoch'] = epoch
    if os.path.exists(path):
        with open(f'{path}/accuracy.json', 'a', encoding='utf-8') as f:
            f.write(',\n')  # 이전 JSON 객체와 구분하기 위해 콤마와 줄바꿈 추가
            json.dump(accuracy, f, indent=4)
    else:
        with open(f'{path}/accuracy.json', 'w', encoding='utf-8') as f:
            json.dump(accuracy, f, indent=4)

def loggingResult(epoch, path, content):
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(path):
        os.makedirs(path)
    
    # 파일 경로 생성
    file_path = os.path.join(path, f'prediction_result_{epoch}.json')
    
    # 파일이 존재하는 경우 append, 그렇지 않으면 새로운 파일 생성
    if os.path.exists(file_path):
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(',\n')  # 이전 JSON 객체와 구분하기 위해 콤마와 줄바꿈 추가
            json.dump(content, f, indent=4)
    else:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=4)

def drawHeatMap(dir_path, epoch, type_list):
    try:
        if(os.path.exists(f"{dir_path}/heatmap") is False):
            os.mkdir(f"{dir_path}/heatmap")
        with open(f"{dir_path}/prediction_result_{epoch}.json", "r") as f:
            data = json.load(f)
        for type in type_list:
            dataframe = pd.DataFrame(data)
            filtered_df = dataframe[dataframe['type'] == type]
    
            if not filtered_df.empty:
                # 피벗 테이블 생성
                heatmap_data = pd.crosstab(filtered_df['prediction'], filtered_df['answer'])

                # 히트맵 그리기
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu")
                plt.title("Heatmap of Predictions vs. Answers")
                plt.xlabel("Answer")
                plt.ylabel("Prediction")

                # 이미지 파일로 저장
                plt.savefig(f"{dir_path}/heatmap/heatmap_predictions_vs_answers_{type}_{epoch}.png")
                plt.close()
    except Exception as e:
        print("에러: ", e)

class ResultCollector():
    def __init__(self):
        self.processing = True
        self.count = 0
        self.prediction_list = []
        self.result_dict = {}

    def end(self):
        self.processing = False

    def update(self, _data, _prediction, _result):
        self.updateCount()
        self.updatePredictList(_data, _prediction, _result)
        self.updateResultDict(_data, _result)

    def updateCount(self):
        self.count = self.count + 1

    def updatePredictList(self, _data, _prediction, _result):
        _data["id"] = self.count
        _data["answer"] = _data['label'].lower()
        _data["prediction"] = _prediction.lower()
        _data["_result"] = _result
        
        self.prediction_list.append(_data)

    def updateResultDict(self, _data, _result_point):
        if(self.result_dict.get(_data['type']) is None):
            self.result_dict[ _data['type'] ] = []
        self.result_dict[ _data['type'] ].append(_result_point)

    def getAccPerCategory(self):
        if(self.processing is False):
            result_dict_accuracy = {}
            for key in list(self.result_dict.keys()):
                result_list_per_category = self.result_dict.get(key)
                result_dict_accuracy[key] = sum(result_list_per_category) / len(result_list_per_category)
            result_dict_accuracy = dict(sorted(result_dict_accuracy.items()))
            print("Result Acc: " , result_dict_accuracy)
            return result_dict_accuracy
        else:
            assert False, "오류: 평가 도중에는 결과를 도출할 수 없음."
        
    def getF1ScorePerCategory(self):
        if(self.processing is False):
            f1_result = getF1Score( list(self.result_dict.keys()), self.prediction_list)
            f1_result = dict(sorted(f1_result.items()))
            print("Result F1: " , f1_result)
            return f1_result
        else:
            assert False, "오류: 평가 도중에는 결과를 도출할 수 없음."


def load_image(image_file, args, input_size=224):
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

def eval_model(_setting, _model , _tokenizer, _image_processor,args):
    image_size = _model.config.force_image_size or _model.config.vision_config.image_size
    question = "<image>\n"+_setting.get("query")

    img_path = _setting.get("image_file")
    if not os.path.exists(img_path):
        print(f"Warning: Image file '{img_path}' does not exist. Skipping this entry.")
        return ""

    loaded_image = load_image(img_path, args, image_size)
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

def inference( _setting, model, tokenizer, _image_processor,args):
    result_collector = ResultCollector()

    annotation_data = None
    with open(_setting['json_path'], "r") as f:
        annotation_data = json.load(f)

    count = 0
    correct = 0
    prediction_list = []
    for data in tqdm(annotation_data):

        if data['type'] == "general_two_option":
            continue

        _setting['image_file'] = os.path.join(_setting['image_dir'], data['image'].replace("./", ""))
        _setting['query'] = data['question']

        prediction = eval_model(_setting, model, tokenizer, _image_processor,args)


        prediction_item = {
            'image' : data['image'],
            'tuning_query' : _setting['query'],
            'tuning_prediction' : prediction,
            'tuning_label' : data['label']
        }

        prediction_list.append(prediction_item)


        if("complex" in data['type']):
            prediction = prediction.lower().strip().replace(".", "")[1:]
        else:
            prediction = prediction.lower().strip().replace(".", "")

        answer = data['label'].lower().strip().replace(".", "")
        prediction = remove_punctuation(prediction)
        answer = remove_punctuation(answer)

        result_point = restrict_compare_Prediction_Answer(prediction, answer)

        result_collector.update(data, prediction, result_point)
        correct += result_point
        count += 1
        if count % 100 == 0:
            print("accuracy: ", round(correct/count, 2)*100 , "%", flush=True)

    with open('./result/result_tuning.json','w') as f:
        json.dump(prediction_list,f,indent = 4)

    result_collector.end()
    result_dict_accuracy = result_collector.getAccPerCategory()
    result_dict_f1 = result_collector.getF1ScorePerCategory()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuning_dir = f"./Tuning_{timestamp}"

    if not os.path.exists(tuning_dir):
        os.makedirs(tuning_dir)

    loggingResult(_setting['epoch'], tuning_dir, result_collector.prediction_list)
    drawHeatMap(tuning_dir, _setting['epoch'], list(result_collector.result_dict.keys()))
    return {"accuracy": result_dict_accuracy, "f1_score": result_dict_f1}

def main():
    
    config = {
        "epoch":"tuning",
        'query': None,
        'conv_mode': None,
        'image_file': None,
        'sep': ',',
        'temperature': 0.2,
        'top_p': None,
        'num_beams': 1,
        'max_new_tokens': 256,
        "model_path": "/data/pretrained/InternVL2-4B",
        "model_base": None,
        "json_path":"/home/EgoOrientBench/all_data/EgocentricDataset/train_benchmark/benchmark.json",
        "image_dir":"/home/EgoOrientBench/all_data/EgocentricDataset"
    }

    model_name = config.get("model_path")
    print(model_name)
    args = Namespace(
        checkpoint=config.get("model_path"),
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
    image_processor = None
    final_result_collection = {}
    
    for i in range(1):
        restrict_seed(i)
        result_dict_accuracy = inference( config , model, tokenizer, image_processor,args)
        print("index:", i, result_dict_accuracy)
        for key in list(result_dict_accuracy.keys()):
            if(final_result_collection.get(key) is None):
                final_result_collection[key] = []
            final_result_collection[key].append(result_dict_accuracy.get(key))
    return final_result_collection

from scipy.stats import t
import numpy as np

def calculateIntervel(data_list):
    sample_mean = np.mean(data_list)
    sample_std = np.std(data_list, ddof=1)
    confidence_level = 0.95
    df = len(data_list) - 1
    t_critical = t.ppf(confidence_level, df)
    lower_bound = sample_mean - t_critical * sample_std / np.sqrt(len(data_list))
    upper_bound = sample_mean + t_critical * sample_std / np.sqrt(len(data_list))
    print(f"{confidence_level}% 신뢰구간: [{round(lower_bound, 5)}, {round(upper_bound, 5)}]", f'{float(np.mean([lower_bound, upper_bound])):.5f}', "±", f'{float(upper_bound-lower_bound)/2:.5f}')
    return f'{float(np.mean([lower_bound, upper_bound])):.5f}', "±", f'{float(upper_bound-lower_bound)/2:.5f}'

if __name__ == '__main__':

    final_result_collection = main()

    accuracy_collect = {}
    for item in final_result_collection['accuracy']:
        for key in item.keys():
            if(accuracy_collect.get(key) is None):
                accuracy_collect[key] = []
            accuracy_collect[key].append( item[key] )
            
    confi_collection = {}

    for a_key in list(accuracy_collect.keys()):
        a_list = accuracy_collect.get(a_key)
        print(a_key, a_list)
        confi_interval = calculateIntervel(a_list)

        confi_collection[a_key] = confi_interval
    with open(f'./result/confidential_interval.json', 'w', encoding='utf-8') as f:
        json.dump(confi_collection, f, indent=4)