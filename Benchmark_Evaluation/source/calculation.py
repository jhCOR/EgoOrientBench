from scipy.stats import t
import numpy as np
from sklearn.metrics import f1_score

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
