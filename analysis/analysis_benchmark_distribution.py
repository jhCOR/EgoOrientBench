import json
import os

if __name__ == '__main__':
    url = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/train_benchmark/benchmark.json")
    with open(url, "r") as f:
        data_list = json.load(f)
    print(len(data_list))

    data_list = [item for item in data_list if item['base_dataset'] == "PACS"]
    total = 0
    data_dict = {}
    for item in data_list:
        if item['type'] == "general_complex":
            total += 1
            original_label = item['original_label']
            if data_dict.get(original_label) is None:
                data_dict[original_label] = []
            
            data_dict[original_label].append(item)

    sum_PERCENTAGE = 0
    for key in data_dict.keys():
        percentage = round(len(data_dict[key])/total, 3)*100
        sum_PERCENTAGE+=percentage
        print(key, percentage, "%")
    print("total:", sum_PERCENTAGE, "%")
