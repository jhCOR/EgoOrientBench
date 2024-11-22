import json
import os
from source.calculate import countCategoryNum, check_duplicate_item, get_category_index
from config import Config

def get_sorted_annotation():
    data = get_annotation_data()
    sorted_data = sorted(data, key=lambda x: x['cateogry'])
    return sorted_data

def get_annotation_data():
    config = Config()
    json_file = config.json_file
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data

def set_annotation_data(annotation_data):
    config = Config()
    json_file = config.json_file
    with open(json_file, 'w') as f:
        json.dump(annotation_data, f, indent=4)

def add_data_to_json(data):
    # JSON 파일 경로 설정
    config = Config()
    json_file = config.json_file
    PREFIX = config.PREFIX

    # JSON 파일이 있는지 확인하고 데이터 추가 또는 새로 생성
    if os.path.exists(json_file):
        # JSON 파일이 이미 존재하는 경우, 데이터를 불러와서 중복 확인 후 추가
        json_data = get_annotation_data()

        if data not in json_data:
            index = get_category_index(data["cateogry"], PREFIX)
            json_data.insert(index, data)
            set_annotation_data(json_data)
            print(f"데이터가 '{json_file}'에 추가되었습니다.")
        else:
            print("중복된 데이터입니다. 추가하지 않았습니다.")
    else:
        # JSON 파일이 없는 경우, 새로운 JSON 파일을 만들고 데이터 추가
        with open(json_file, 'w') as f:
            json.dump([data], f, indent=4)
        print(f"새로운 JSON 파일 '{json_file}'이 생성되었고, 데이터가 추가되었습니다.")