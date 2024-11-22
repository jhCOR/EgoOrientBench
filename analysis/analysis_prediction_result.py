import json
import datetime

alphabet_mapper = {"a":"front", "b": "front right", "c": "right", "d": "back right", "e": "back", "f": "back left", "g": "left", "h": "front left"}
task_mapper = { "overall":"overall", "general_complex":"choose", "general_binary":"verify", "freeform":"freeform" }

def contains_toy_keywords(path):
    keywords = ['dinosaur', 'doll', 'teddy', 'toy_animals', 'toy_boat', 'toy_bus',
                'toy_car', 'toy_motorcycle', 'toy_plane', 'toy_train', 'toy_truck']
    return any(keyword in path for keyword in keywords)

def contains_domainnet_keywords(path):
    keywords = ['clipart', 'painting', 'real', 'sketch']
    return any(keyword in path for keyword in keywords)

def get_data_source(path):
    if path.split("/")[-1][0] == "n":
        return "imagenet"
    elif "pacs" in path.split("/")[-1]:
        return "pacs"
    elif contains_toy_keywords(path):
        return "3D"
    elif contains_domainnet_keywords(path):
        return "DomainNet"
    elif "COCO" in path.split("/")[-1]:
        return "D3_Eval"
    else:
        return "D3_Eval"
    
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

def calculate_accuracy(data_list, question_type):

    key_of_result = "result"
    total_result = {"general_complex": [], "general_binary": [], "freeform": []}

    # 데이터를 question type별로 분류
    for item in data_list:
        total_result[item['type']].append(item)

    total_result_acc = {}
    for key, images in total_result.items():
        correct_count = 0
        for image in images:
            if key == "general_complex":
                image['_result'] = compare_Prediction_Answer(image['prediction'], image['answer'])
            if key == "general_complex" and len(image['prediction']) == 1:
                image['prediction'] = alphabet_mapper.get(image['prediction'])
                if image['prediction'] == image['answer']:
                    image[key_of_result] = 1
            correct_count += image.get(key_of_result, 0)

        accuracy = round(correct_count / len(images), 3) * 100 if images else 0
        print(f"{key}: {len(images)} images, {accuracy}% accuracy")
        total_result_acc[key] = accuracy
    print("================================")
    return total_result_acc

def calculate_type_accuracy(data_list, question_type):
    key_of_result = "result"
    """특정 question type에 대해 데이터 소스별로 정확도 계산 및 출력."""
    print(f"\n<{task_mapper.get(question_type)}>")
    data_dict = {}

    # 특정 question type의 데이터를 소스별로 분류
    for item in data_list:
        if item['type'] == question_type:
            source = get_data_source(item['image'])
            if source == "others":
                print(item['image'])
            data_dict.setdefault(source, [])

            # 예측값과 정답 비교
            if question_type == "general_complex":
                item['_result'] = compare_Prediction_Answer(item['prediction'], item['answer'])
            if len(item['prediction']) == 1:
                item['prediction'] = alphabet_mapper.get(item['prediction'])
                if item['prediction'] == item['answer']:
                    item[key_of_result] = 1
            data_dict[source].append(item)
    
    result_per_dataset = {}
    for source, images in data_dict.items():
        print(len(images))
        correct_count = sum(image.get(key_of_result, 0) for image in images)
        accuracy = round(correct_count / len(images), 3) * 100 if images else 0

        print(f"{source}: {len(images)} images, {accuracy}% accuracy")
        result_per_dataset[source] = accuracy
    return result_per_dataset

if __name__ == '__main__':

    url_list = [
    ]
    final_score = {}
    for url in url_list:
    
        # 파일 열기 및 데이터 로드
        with open(url, "r") as f:
            data_list = json.load(f)

        overall_score = calculate_accuracy(data_list, question_type="general_two_option")
        choose_score = calculate_type_accuracy(data_list, "general_complex")
        verify_score = calculate_type_accuracy(data_list, "general_binary")
        freeform_score = calculate_type_accuracy(data_list, "freeform")
        print(freeform_score)
        final_score[url] = {"overall": overall_score, "choose": choose_score, "verify": verify_score, "freeform": freeform_score}
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    averave_score = {}
    for key in final_score.keys():
        print(f"\n{key}")
        for task, score in final_score[key].items():
            print(f"{task}: {score}")
            averave_score.setdefault(task, {})

            for metric, acc in score.items():
                averave_score[task].setdefault(metric, 0)
                averave_score[task][metric] += acc
    
    for task, score in averave_score.items():
        for metric, acc in score.items():
            averave_score[task][metric] = round(acc / len(url_list), 1)
    print("\naverave_score")
    print(averave_score)