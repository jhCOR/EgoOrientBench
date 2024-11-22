import random
import json
import argparse
import os
import copy

CONST_uniform = "uniform"
dir_to_dir_map = {
  "front right":"facing right while facing the camera",
  "front left" : "facing left while facing the camera",
  "front": "facing the camera",
  "left" : "facing left",
  "right": "facing right",
  "back" : "facing away the camera",
  "back left": "toward left while facing away the camera",
  "back right": "toward right while facing away the camera"
}

def checkDataNumperCategory(_data_raw):
    _data = copy.deepcopy(_data_raw)
    cateory_count = {}
    for item in _data:
        if(cateory_count.get(item['direction'])) is None:
            cateory_count[item['direction']] = []
        cateory_count[item['direction']].append(item)

    most_small_category_num = 100000      
    for key in cateory_count.keys():
        amount = len(cateory_count.get(key))
        if(most_small_category_num> amount ):
            most_small_category_num = amount
        print( key, amount )
    return cateory_count, most_small_category_num

def shuffleTwo(answer, not_answer):
  random_list = [answer, not_answer]
  random.shuffle(random_list)
  return random_list

def alhpabetWithDirection(label):
    list_of_dir = ["front", "front right", "right", "back right", "back", "back left", "left", "front left"]
    list_of_alphabet = ["A.", "B.", "C.", "D.", "E.","F.", "G.","H."]

    joined_list = [ list_of_alphabet[num]+list_of_dir[num] for num in range(len(list_of_alphabet)) ]

    index = list_of_dir.index(label)
    label = {"label_1":label, 
             "label_2": list_of_alphabet[index].replace(".", ""), 
             "label_3":joined_list[index]}
    return " ".join(joined_list), label

def alhpabetWithVerboseDirection(label):
    list_of_dir = ["front", "front right", "right", "back right", "back", "back left", "left", "front left"]

    list_of_verbose_dir = [dir_to_dir_map.get(dir) for dir in list_of_dir]
    list_of_alphabet = ["A.", "B.", "C.", "D.", "E.","F.", "G.","H."]

    joined_list = [ "\n"+list_of_alphabet[num]+list_of_verbose_dir[num] for num in range(len(list_of_alphabet)) ]

    index = list_of_dir.index(label)
    label = {"label_1":list_of_verbose_dir[index], 
             "label_2": list_of_alphabet[index].replace(".", ""), 
             "label_3":joined_list[index]}
    return " ".join(joined_list), label

class TemplateClass():
    def __init__(self, item, key):
        self.item = item
        self.key = key
        self.category_name = self.item['category_name'].split(",")[0]
        self.step_list = []

    def type_A_question(self):
        sen, label = alhpabetWithDirection(self.key)
        prompt = {
                "type":"general_complex",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": self.item['path'],
                "original_label":self.item['direction'],
                "question": f"From the perspective of the camera, which orientation is the {self.category_name} in the photo facing? "+ sen +
                f"\n Answer with the option's letter and word from the given choices directly. Answer: [choose one in {sen}]",
                "label":label['label_1'],
                "category_name":self.item['category_name']
            }
        self.step_list.append(prompt)
        return self

    def type_B_question(self):
        ver_sen, ver_label = alhpabetWithVerboseDirection(self.key)
        prompt = {
                "type":"freeform",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": self.item['path'],
                "original_label":self.item['direction'],
                "question": f"From the perspective of the camera, Answer what orientation the {self.category_name} in the picture is facing.",
                "label" : ver_label['label_1'],
                "category_name":self.item['category_name']
            }
        self.step_list.append(prompt)
        return self

    def type_C_question(self):
        other_direction_list = list(dir_to_dir_map.keys())
        other_direction_list.remove(self.item['direction'])
        shuffled_list = shuffleTwo(self.item['direction'], random.choice(other_direction_list))
        prompt = {
                "type":"general_two_option",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": self.item['path'],
                "original_label": self.item['direction'],
                "question": f"Is the {self.category_name} facing '{shuffled_list[0]}' or '{shuffled_list[1]}' from the camera's perspective? \nAnswer the question using a single word or phrase only.",
                "label": self.item['direction'],
                "category_name": self.item['category_name']
            }
        self.step_list.append(prompt)
        return self

    def type_D_question(self, other_dir):
        positive_image = self.item
        p_category_name = positive_image['category_name'].split(",")[0]
        direction = positive_image['direction']
        positive_prompt = {
                "type":"general_binary",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": positive_image['path'],
                "original_label":positive_image['direction'],
                "question": f"Is the {p_category_name} facing '{direction}' from the camera's perspective? \nAnswer with 'yes' or 'no' only.",
                "label":"yes",
                "category_name":p_category_name
            }
        negative_prompt = {
                "type":"general_binary",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": positive_image['path'],
                "original_label":positive_image['direction'],
                "question": f"Is the {p_category_name} facing '{other_dir}' from the camera's perspective? \nAnswer with 'yes' or 'no' only.",
                "label":"no",
                "category_name":p_category_name
            }
        self.step_list.append(positive_prompt)
        self.step_list.append(negative_prompt)
        return self

    def type_E_question(self, positive_image, negative_image):
        p_category_name = positive_image['category_name'].split(",")[0]
        n_category_name = negative_image['category_name'].split(",")[0]
        direction = positive_image['direction']
        positive_prompt = {
                "type": "verbose_binary",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": positive_image['path'],
                "original_label":positive_image['direction'],
                "question": f"Is the {p_category_name} facing '{dir_to_dir_map.get(direction)}' from the camera's perspective? \nAnswer with 'yes' or 'no' only",
                "label":"yes",
                "category_name":p_category_name
            }
        negative_prompt = {
                "type":"verbose_binary",
                "domain":self.item['domain'],
                "base_dataset":self.item['base_dataset'],
                "image": negative_image['path'],
                "original_label":negative_image['direction'],
                "question": f"Is the {n_category_name} facing '{dir_to_dir_map.get(direction)}' from the camera's perspective? \nAnswer with 'yes' or 'no' only",
                "label":"no",
                "category_name":n_category_name
            }
        self.step_list.append(positive_prompt)
        self.step_list.append(negative_prompt)
        return self

    def end(self):
        return self.step_list

def list_subfolders(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return subfolders

def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="./dataset/Source/raw_dataset_example.json")
    parser.add_argument("--output_path", type=str, default="./dataset/Result")
    parser.add_argument("--option", type=str, default="appending") #appending
    parser.add_argument("--distribution", type=str, default="None") #uniform
    parser.add_argument("--mplugstyle", type=bool, default=False)
    args = parser.parse_args() 

    print(list_files("../"))

    with open(args.json_path) as f:
        annotation = json.load(f)

    list_per_domain = {}
    for item in annotation:
        if list_per_domain.get(item['base_dataset']) is None:
            list_per_domain[item['base_dataset']] = []

        list_per_domain[item['base_dataset']].append(item)
    
    valid_data_collection = []
    for key in list_per_domain.keys():
        for item in list_per_domain.get(key):

            templateHandler = TemplateClass(item, item['direction'])
            templateHandler.type_A_question().type_B_question().type_C_question()

            negative_dir_list = [ _dir for _dir in list(dir_to_dir_map.keys()) if _dir != key ]
            random_other_dir = random.choice(negative_dir_list)

            question_list = templateHandler.type_D_question(random_other_dir).end()

            valid_data_collection = valid_data_collection + question_list
    
    for item in valid_data_collection:
        assert item.get("domain") is not None, "에러"
        assert item.get("base_dataset") is not None, "에러"
    
    print("clean")
    print(len(valid_data_collection))

    assert str(args.output_path)[-1] != "/", "Path should be end without /"

    if os.path.exists(args.output_path) is False:
        os.makedirs(args.output_path, exist_ok=True)
    with open(f"{args.output_path}/all_other_benchmark.json", 'w') as f:
        json.dump(valid_data_collection, f, indent=4)