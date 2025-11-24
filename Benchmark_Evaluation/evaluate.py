from datetime import datetime
from tqdm import tqdm
import argparse
import json
import os

from source.calculation import calculateIntervel, restrict_compare_Prediction_Answer, compare_Prediction_Answer
from source.utils import remove_punctuation
from source.logger import drawHeatMap, loggingResult
from source.ResultCollector import ResultCollector

from source.Wrapper.main import create_model_wrapper

def inference( setting, model):
    result_collector = ResultCollector()

    annotation_data = None
    with open(setting['json_path'], "r") as f:
        annotation_data = json.load(f)
    
    if setting['mode'] == "test":
        annotation_data = annotation_data[:5]
    elif setting['mode'] == "check":
        annotation_data = annotation_data[:100]
    else:
        print("전체 데이터 사용")
    
    print("데이터 길이: ", len(annotation_data))

    for data in tqdm(annotation_data):
        image_path = os.path.join(setting['image_dir'], data['image'].split("/")[-1])
        prompt = data['question']

        prediction = model.predict(prompt, image_path)

        if("choose" in data['type']):
            if len(prediction) > 1:
                prediction = prediction.lower().strip().replace(".", "")[1:]
            else:
                prediction = prediction.lower().strip().replace(".", "")
        else:
            prediction = prediction.lower().strip().replace(".", "")

        answer = data['label'].lower().strip().replace(".", "")
        prediction = remove_punctuation(prediction)
        answer = remove_punctuation(answer)

        if setting['restrict_eval'] == "yes":
            result_point = restrict_compare_Prediction_Answer(prediction, answer)
        else:
            result_point = compare_Prediction_Answer(prediction, answer)
        
        result_collector.update(data, prediction, result_point)
        result_collector.intermediate_save(setting['dir_name'])

    result_collector.end()
    result_dict_accuracy = result_collector.getAccPerCategory()
    result_dict_f1 = result_collector.getF1ScorePerCategory()

    loggingResult(setting['mode'], f"{setting['dir_name']}", result_collector.prediction_list)
    drawHeatMap(f"{setting['dir_name']}", setting['mode'], list(result_collector.result_dict.keys()))
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

def main(config, model):

    final_result_collection = {}
    for i in range(1):
        result_dict_accuracy = inference( config, model )
        print("index:", i, result_dict_accuracy)

        for key in result_dict_accuracy.keys():
            if(final_result_collection.get(key) is None):
                final_result_collection[key] = []
            final_result_collection[key].append(result_dict_accuracy.get(key))
    return final_result_collection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="../all_data/EgocentricDataset/benchmark_data/benchmark.json")
    parser.add_argument("--image_dir", type=str, default="../all_data/EgocentricDataset/imagenet_after")

    parser.add_argument("--model_type", type=str, default="opensource")
    parser.add_argument("--model_name", type=str, default="llava")
    
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--key_path", type=str, default=None)

    parser.add_argument("--mode", type=str, default="inference")
    parser.add_argument("--restrict_eval", type=str, default="yes")
    args = parser.parse_args()

    now = datetime.now()

    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
    }
    model_setting = {
        "checkpoint": args.checkpoint,
        "model_path": args.model_path,
        "model_base": args.model_base,
    }
    model_wrapper = create_model_wrapper(args.model_type, 
                                         args.model_name,
                                           configuration, 
                                           model_setting=model_setting,
                                           key=args.key,
                                           key_path=args.key_path)

    try:
        config = {
            "json_path": args.json_path,
            "image_dir": args.image_dir,
            "dir_name": f"./_Result/benchmark_{args.model_name}_{now.strftime('%H%M%S')}",
            "mode": args.mode,
            "restrict_eval": args.restrict_eval
        }
        if config["restrict_eval"] == "yes":
            print("The code pipeline will only mark answers as correct if it selects the exact right answer.")
        else:
            print("The code pipeline will mark answers as correct if the correct answer is included within a sentence or short phrase.")
        
        print("파일 경로 확인:", os.path.exists(f'./{config["dir_name"]}'))
        
        os.makedirs(f'{config["dir_name"]}', exist_ok=True)
        final_result_collection = main(config, model_wrapper)
        postprocess_result(final_result_collection, config)
    except Exception as e:
        print("에러: ", e)

    
