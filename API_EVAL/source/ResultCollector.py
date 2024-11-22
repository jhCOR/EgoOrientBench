import json
from datetime import datetime
from source.calculation import getF1Score

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

    def intermediate_save(self, path):
        now = datetime.now()
        with open(f"{path}/intermediate.json", 'w') as f:
            json.dump(self.prediction_list, f, indent=4)
    
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
