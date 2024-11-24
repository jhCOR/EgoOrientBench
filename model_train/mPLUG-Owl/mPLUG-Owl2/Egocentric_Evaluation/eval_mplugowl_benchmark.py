import torch
from PIL import Image
from transformers import TextStreamer
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from PIL import Image
import torch
import json
import os
import random 
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from scipy.stats import t
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
    torch.backends.cudnn.benchmark = False

def remove_punctuation(text):
    # string.punctuation에는 모든 문장부호가 포함되어 있습니다.
    return text.translate(str.maketrans('', '', string.punctuation))

def image_parser(_setting):
    out = _setting.get("image_file").split( _setting.get("sep") )
    return out

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
        _data["prediction"] = _prediction.lower()
        _data["answer"] = _data['label'].lower()
        _data["result"] = _result
        
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

def inference( _setting, model, tokenizer, _image_processor):
    result_collector = ResultCollector()

    annotation_data = None
    with open(_setting['json_path'], "r") as f:
        annotation_data = json.load(f)
    random.shuffle(annotation_data)

    for data in tqdm(annotation_data):
        _setting['image_file'] = os.path.join(_setting['image_dir'], data['image'].replace("./", ""))
        _setting['query'] = data['question'].replace("<image>\n ", "")
        _setting['query'] = data['question'].replace("<|image|>\n ", "")
        
        prediction = eval_model(_setting, model, tokenizer, _image_processor)
        prediction = prediction.replace("</s>", "")
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
        print(prediction, answer)    
        result_point = None
        result_point = restrict_compare_Prediction_Answer(prediction, answer)

        result_collector.update(data, prediction, result_point)

    result_collector.end()
    result_dict_accuracy = result_collector.getAccPerCategory()
    result_dict_f1 = result_collector.getF1ScorePerCategory()

    loggingResult("zeroshot", "./Zeroshot3", result_collector.prediction_list)
    drawHeatMap("./Zeroshot3", "zeroshot", list(result_collector.result_dict.keys()))
    return {"accuracy": result_dict_accuracy, "f1_score": result_dict_f1}

def main():
    
    model_name = "models--MAGAer13--mplug-owl2-llama2-7b"
    model_path = 'MAGAer13/mplug-owl2-llama2-7b'
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

    final_result_collection = {}
    config = {
            'query': None,
            'conv_mode': None, 
            'image_file': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256,
            "json_path": os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/benchmark.json"),
            'image_dir': os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/"),
        }
    
    for i in range(1):
        restrict_seed(i)
        result_dict_accuracy = inference( config , model, tokenizer, image_processor)
        print("index:", i, result_dict_accuracy)
        for key in list(result_dict_accuracy.keys()):
            if(final_result_collection.get(key) is None):
                final_result_collection[key] = []
            final_result_collection[key].append(result_dict_accuracy.get(key))
    return final_result_collection


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