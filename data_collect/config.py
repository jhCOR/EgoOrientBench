from dataclasses import dataclass
import os
import json

def getCategoryMap(category_annotation_path):
    assert os.path.exists(category_annotation_path)
    category_mapped_data = None
    with open(category_annotation_path, 'r') as f:
        category_mapped_data = json.load(f)
    return category_mapped_data

def category_list(manual_setting, category_list, prefix=None):
    if manual_setting is True:
        return

def get_folder_list(path):
    folders = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders

@dataclass
class Config:
    PREFIX: str = "../../../../../data/IMAGE_NET/ILSVRC/Data/CLS-LOC/train/"
    category_list = get_folder_list(PREFIX)
    after_path: str = "./static/imagenet_after/"
    additional_after_path= None
    json_file:str = './static/annotation.json'
    memory_path:str = "./system_memory/system_memory.json"
    dict_table = {2:"front", 1: "front left", 3: "front right",
              4: "left", 6: "right",
              8: "back", 7: "back left", 9: "back right"}
    category_map = getCategoryMap("./static/category_mapping.json")
