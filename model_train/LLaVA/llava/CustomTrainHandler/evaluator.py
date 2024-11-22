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
    get_model_name_from_path,
)

from transformers import TrainerCallback, TrainingArguments
from sklearn.metrics import f1_score
from io import BytesIO
from tqdm import tqdm
from PIL import Image
import argparse
import requests
import torch
import json
import re
import os
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string

def remove_punctuation(text):
    # string.punctuation에는 모든 문장부호가 포함되어 있습니다.
    return text.translate(str.maketrans('', '', string.punctuation))

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

def eval_model(_setting, _model , _tokenizer, _image_processor, model_name):
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
    ).to(_model.device, dtype=torch.float16)

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

def compare_Prediction_Answer(prediction, answer):
    two_word_label = ["front right", "front left", "back left", "back right"]
    result = (answer in prediction)

    if( len( answer.split(" ")) == 1 ):
        for twl in two_word_label:
            if( twl in prediction):
                result = False
                break
    if(result):
        point = 1
    else:
        point = 0
    return point

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
        format = {
            "id":self.count, "image":_data['image'],
            "type":_data['type'], "question":_data['question'], 
            "prediction":_prediction.lower(), "answer":_data['label'].lower(), 
            "result":_result, "original_label":_data['original_label'],
            "base_dataset":_data['base_dataset'], "domain":_data['domain']
        }
        self.prediction_list.append(format)

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

def external_inference(arg, eval_dataset_path, model, tokenizer, _image_processor):
        _setting = {
            'query': None,
            'conv_mode': None, 
            'image_file': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256, 
            "epoch":"last"
        }
        result_collector = ResultCollector()

        annotation_data = None
        with open(eval_dataset_path, "r") as f:
            annotation_data = json.load(f)

        dir_path = os.path.dirname(eval_dataset_path)
        for data in tqdm(annotation_data):
            
            _setting['image_file'] = os.path.join(dir_path, data['image'].replace("./", ""))
            _setting['query'] = data['question'].replace("<image>\n ", "")

            prediction = eval_model(_setting, model, tokenizer, _image_processor, tokenizer.name_or_path)
            if("complex" in data['type']):
                prediction = prediction.lower().strip().replace(".", "")[1:]
            else:
                prediction = prediction.lower().strip().replace(".", "")
            answer = data['label'].lower().strip().replace(".", "")
            prediction = remove_punctuation(prediction)
            answer = remove_punctuation(answer)
            
            result_point = restrict_compare_Prediction_Answer(prediction, answer)
            result_collector.update(data, prediction, result_point)

        result_collector.end()
        result_dict_accuracy = result_collector.getAccPerCategory()
        result_dict_f1 = result_collector.getF1ScorePerCategory()
        
        loggingResult(_setting['epoch'], arg['result_output_dir'], result_collector.prediction_list)
        drawHeatMap( arg['result_output_dir'], _setting['epoch'], list(result_collector.result_dict.keys()))
        return {"accuracy": result_dict_accuracy, "f1_score": result_dict_f1}

class QAEvaluationCallback(TrainerCallback):
    def __init__(self, arg, eval_dataset_path, eval_steps=50):
        self.eval_steps = eval_steps
        self.setting = self.makeSetting(arg)
        self.log_dir = arg['result_output_dir']
        self.eval_dataset_path = eval_dataset_path

    def makeSetting(self, argument):
        return {
            'query': None,
            'conv_mode': None, 
            'image_file': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256, 
        }

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        if args.evaluation_strategy == "epoch" and (args.num_train_epochs == round(state.epoch)):
            image_processor = kwargs['model'].get_vision_tower().image_processor
            self.setting['epoch'] = state.epoch
            result_list = self.inference(self.setting, kwargs['model'], kwargs['tokenizer'], image_processor)
            
            print("Epoch end accuracy: ", result_list)
            metrics = {
                "eval_accuracy": result_list['accuracy'],
                "epoch": state.epoch
            }
            logging(self.log_dir, state.epoch, result_list)

    def on_step_end(self, args: TrainingArguments, state, control, **kwargs):
        if args.evaluation_strategy == "steps" and state.global_step % self.eval_steps == 0:
            image_processor = kwargs['model'].get_vision_tower().image_processor
            self.setting['epoch'] = state.global_step
            result_list = self.inference(self.setting, kwargs['model'], kwargs['tokenizer'], image_processor)
            
            print("Step end accuracy: ", result_list)
            metrics = {
                "eval_accuracy": result_list['accuracy'],
                "epoch": state.epoch
            }
            logging(self.log_dir, state.epoch, result_list)

    def inference(self, _setting, model, tokenizer, _image_processor):
        result_collector = ResultCollector()

        annotation_data = None
        with open(self.eval_dataset_path, "r") as f:
            annotation_data = json.load(f)
        dir_path = os.path.dirname(os.path.dirname(self.eval_dataset_path))
        for data in tqdm(annotation_data):
            
            _setting['image_file'] = os.path.join(dir_path, data['image'].replace("./", ""))
            _setting['query'] = data['question'].replace("<image>\n ", "")

            prediction = eval_model(_setting, model, tokenizer, _image_processor, tokenizer.name_or_path)
            if("complex" in data['type']):
                prediction = prediction.lower().strip().replace(".", "")[1:]
            else:
                prediction = prediction.lower().strip().replace(".", "")
            answer = data['label'].lower().strip().replace(".", "")
            prediction = remove_punctuation(prediction)
            answer = remove_punctuation(answer)
            
            result_point = restrict_compare_Prediction_Answer(prediction, answer)

            result_collector.update(data, prediction, result_point)

        result_collector.end()
        result_dict_accuracy = result_collector.getAccPerCategory()
        result_dict_f1 = result_collector.getF1ScorePerCategory()
        
        loggingResult(_setting['epoch'], self.log_dir, result_collector.prediction_list)
        drawHeatMap( self.log_dir, _setting['epoch'], list(result_collector.result_dict.keys()))
        return {"accuracy": result_dict_accuracy, "f1_score": result_dict_f1}