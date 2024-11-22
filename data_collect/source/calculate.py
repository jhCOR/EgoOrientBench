import json
from source.data_info import get_category_per_image_count
from config import Config

def get_category_index(category, prefix):
    config = Config()
    category_per_count = get_category_per_image_count()

    category_list = config.category_list
    sorted_category_list = sorted(category_list)

    index = 0
    for cate in sorted_category_list:
        if category == cate:
            break
        else:
            index += category_per_count[cate]

    return index

def countCategoryNum(PREFIX):
    config = Config()
    json_file = config.json_file
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    category_count = {}
    category_list = config.category_list
    for item in category_list:
        category_count[item] = 0
    
    for item in json_data:
        cate = item['cateogry']
        if category_count.get(cate) is not None:
            category_count[cate] += 1
    return category_count

def check_duplicate_item(json_data, path):
    for item in json_data:
        if item['path'] == path:
            return item
    return None

