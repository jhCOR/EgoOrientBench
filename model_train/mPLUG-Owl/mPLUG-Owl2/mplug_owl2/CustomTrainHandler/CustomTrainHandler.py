import os
from mplug_owl2.CustomTrainHandler.calculate import getNextTrialNum
import json
import copy

def getOutputDirPath(_dir, _name="", prefix="try_"):
    print("CustomTrainManager가 결과를 수집할 디렉터리를 결정합니다. ")
    if(os.path.exists(_dir) is False):
        os.mkdir(_dir)
    current_output_path_dir = None
    if(len(_name)<1):
        current_trial_num = getNextTrialNum(_dir)
        current_output_file_name = prefix + str(current_trial_num)
    else:
        current_output_file_name = _name
    current_output_path_dir = _dir + "/" + current_output_file_name
    print(f"CustomTrainManager가 다음의 위치에 결과를 수집합니다. {current_output_path_dir}")
    if(os.path.exists(current_output_path_dir) is False):
        os.mkdir(current_output_path_dir)
    return current_output_path_dir

def saveAllArgument(path, model_args, data_args, training_args):
    print("CustomTrainManager가 모든 인자들을 저장합니다. ")
    # 파싱된 인수를 딕셔너리로 변환
    args_dict = {
        "model_args": copy.deepcopy(vars(model_args)),
        "data_args": copy.deepcopy(vars(data_args)),
        "training_args": copy.deepcopy(vars(training_args))
    }
    # JSON 형식으로 파일에 저장
    with open(f"{path}/configuration.json", "w") as f:
        json.dump(str(args_dict), f, indent=4)
    
def saveAllResult(path, result_data):
    serialized_argument = str(result_data)
    with open(path + "/result.json", "w") as f:
        json.dump(serialized_argument, f, indent=4)
