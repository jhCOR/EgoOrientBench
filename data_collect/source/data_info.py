import os
import json
from config import Config

def get_folder_list(path):
    folders = []
    for item in os.listdir(path):
        if os.path.isdir(os.path.join(path, item)):
            folders.append(item)
    return folders

def list_files_in_directory(directory):
    files_list = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        
        if os.path.isfile(file_path):
            files_list.append(file)

    return files_list

def get_category_per_image_count():
    config = Config()
    memory_path = config.memory_path
    with open(memory_path, 'r') as f:
        system_memory = json.load(f)
    return system_memory

def set_category_per_image_count(category, increase = False, decrease = False):
    config = Config()
    memory_path = config.memory_path
    
    try:
        with open(memory_path, 'r') as f:
            system_memory = json.load(f)
            if increase is True:
                system_memory[category] += 1
            elif decrease is True:
                system_memory[category] -= 1

            with open(memory_path, 'w') as f:
                json.dump(system_memory, f, indent=4)
        return True
    except Exception as e:
        print(e)
        return False