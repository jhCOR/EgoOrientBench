import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def logging(path, epoch, accuracy):
    accuracy['epoch'] = epoch
    if os.path.exists(path):
        with open(f'{path}/accuracy.json', 'a', encoding='utf-8') as f:
            f.write(',\n')
            json.dump(accuracy, f, indent=4)
    else:
        with open(f'{path}/accuracy.json', 'w', encoding='utf-8') as f:
            json.dump(accuracy, f, indent=4)

def loggingResult(epoch, path, content):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, f'prediction_result_{epoch}.json')
    
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