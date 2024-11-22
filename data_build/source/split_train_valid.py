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
    
def getHardCodingQuestion(direction, category, index):
    sen, label = alhpabetWithDirection(direction)
    category = category.split(",")[0]
    question_choose = f"From the perspective of the camera, which orientation is the {category} in the photo facing? {sen} \n Answer with the option's letter and word from the given choices directly."
    answer_choose = label['label_3']

    if( (index%2) == 0):
        question_verify =  f"Is the {category} facing '{direction}' from the camera's perspective? \nAnswer with 'yes' or 'no' only."
        answer_verify = "yes"
    else:
        other_direction_list = list(dir_to_dir_map.keys())
        other_direction_list.remove(direction)
        other_dir = random.choice(other_direction_list)
        question_verify =  f"Is the {category} facing '{other_dir}' from the camera's perspective? \nAnswer with 'yes' or 'no' only."
        answer_verify = "no"

    other_direction_list = list(dir_to_dir_map.keys())
    other_direction_list.remove(direction)
    shuffled_list = shuffleTwo(direction, random.choice(other_direction_list))

    question_compare = f"Is the {category} facing '{shuffled_list[0]}' or '{shuffled_list[1]}' from the camera's perspective? \nAnswer the question using a single word or phrase only."
    answer_compare = direction
    
    question_list = [{"from":"human", "value":question_choose}, {"from":"gpt", "value":answer_choose}, 
                     {"from":"human", "value":question_verify}, {"from":"gpt", "value":answer_verify},
                     {"from":"human", "value":question_compare}, {"from":"gpt", "value":answer_compare}]
    return question_list

class DataBuilder():
    def __init__(self, annotation_data):
        self.__annotations__ = annotation_data
        self.key_list = self.__annotations__.keys()
        self.valid_data_collection = []

    def getNSample(self, _key, NUM = 20):
        print("키: ", _key, "/ 데이터 길이: ", len(self.__annotations__.get( _key )), "샘플링개수: ", NUM)
        list_per_key = self.__annotations__.get( _key )

        temperate_list = {}
        for _item in list_per_key:
            if(temperate_list.get(_item['image']) is None):
                temperate_list[ _item['image'] ] = []
            temperate_list[ _item['image'] ].append(_item)

        key_list = list(temperate_list.keys())
        random.shuffle(key_list)

        valid_target_list = key_list[:NUM]
        train_target_list = key_list[NUM:]

        valid_data = [temperate_list[key][0] for key in valid_target_list]
        train_data = []
        for sub_key in train_target_list:
            train_data = train_data + temperate_list[sub_key]
        self.__annotations__[_key] = train_data

        print("남은 훈련 데이터 길이: ", len(self.__annotations__.get( _key )))

        self.checkLeakage(train_data, valid_data, "데이터 누출 검사") #데이터 누출 검출
        self.checkDuplicate(valid_data, "데이터 중복 검사") #데이터 중복을 허용하지 않음
        return valid_data
    
    def checkLeakage(self, original, sampled, name=None, verbose=True):
        original_id = [_row['image'] for _row in original]
        for _item in sampled:
            assert _item['image'] not in original_id
        original_id = None
        if(verbose):
            print(name, "성공 ", "데이터 중복이 발견되지 않았습니다.")

    def checkDuplicate(self, data_list, name=None):
        _tem = []
        for data in data_list:
            assert data not in _tem
            _tem.append(data)
        _tem = None
        print(name, "성공 ", "데이터 중복이 발견되지 않았습니다.")
    
    def build(self, num_per=20, option="None", option_uniform=None):
        for key in self.key_list:
            sampled_valid_datas = self.getNSample(key, num_per)
            for item in sampled_valid_datas:
                assert item not in self.__annotations__[key]
                
                templateHandler = TemplateClass(item, key, self.key_list)
                templateHandler.type_A_question().type_B_question().type_C_question()
                
                negative_dir_list = [ _dir for _dir in list(self.key_list) if _dir != key ]
                random_other_dir = random.choice(negative_dir_list)

                question_list = templateHandler.type_D_question(random_other_dir).end()
                self.valid_data_collection = self.valid_data_collection + question_list
        
        if(option == "appending"):
            print("<----각 스타일 질문 뒤에 자동으로 complex, binary QA쌍을 덧붙입니다.---->")
            minimun = getMinimunLength(self.__annotations__)
            for key in self.key_list:
                data_list = self.__annotations__.get(key)

                if(option_uniform > 0):
                    print("option_uniform: ", minimun)
                    self.__annotations__[key] = data_list[:minimun]
                    data_list = self.__annotations__.get(key)
                print(key, "len(data_list): ", len(data_list))

                for index in range(len(data_list)):
                    data = data_list[index]
                    data['conversations'] = [ data['conversations'][0], data['conversations'][1] ] + getHardCodingQuestion(data['direction'], data['category_name'], index)
        return self.__annotations__, self.valid_data_collection

def getMinimunLength(data_dict):
    key_list = data_dict.keys()
    minimun = len( data_dict.get( random.choice(list(key_list)) ) )
    for key in key_list:
        if( len( data_dict.get(key) ) <minimun):
            minimun=len( data_dict.get(key))
    return minimun

class TemplateClass():
    def __init__(self, item, key, key_list):
        self.item = item
        self.key = key
        self.key_list = key_list
        self.category_name = self.item['category_name'].split(",")[0]
        self.step_list = []

    def type_A_question(self):
        sen, label = alhpabetWithDirection(self.key)
        prompt = {
                "type":"general_complex",
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
                "image": positive_image['path'],
                "original_label":positive_image['direction'],
                "question": f"Is the {p_category_name} facing '{direction}' from the camera's perspective? \nAnswer with 'yes' or 'no' only.",
                "label":"yes",
                "category_name":p_category_name
            }
        negative_prompt = {
                "type":"general_binary",
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
                "image": positive_image['path'],
                "original_label":positive_image['direction'],
                "question": f"Is the {p_category_name} facing '{dir_to_dir_map.get(direction)}' from the camera's perspective? \nAnswer with 'yes' or 'no' only",
                "label":"yes",
                "category_name":p_category_name
            }
        negative_prompt = {
                "type":"verbose_binary",
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

def saveToFile(data, path, _type, subfix=""):
    if(_type == "train"):
        data_collection = []
        for key in data.keys():
            data_collection = data_collection + data.get(key)
    elif(_type == "valid"):
        data_collection = data

    print("데이터 개수: ", len(data_collection))
    num = path.split("/")[-1].split("_")[-1]
    if os.path.exists(path) is False:
        os.makedirs(path, exist_ok=True)
    with open(f"{path}/imagenet_13b_{_type}_ver_{num}_{subfix}.json", 'w') as f:
        json.dump(data_collection, f, indent=4)

def split_data(dir_path, annotation_data, option, distribution, mplugStyle=False):
    if(mplugStyle):
        print("THIS is mplug-owl STYLE!!")
        for _data_row in annotation_data:
            _data_row['conversations'][0]['value'] = _data_row['conversations'][0]['value'].replace("<image>", "<|image|>")
    
    print(len(annotation_data))
    category_sorted_dict, minimun_amount = checkDataNumperCategory(annotation_data)
    data_builder = DataBuilder(category_sorted_dict)
    if(len(annotation_data)<=6000):
        num_per = 10
    else:
        num_per = 50
    
    print("카테고리당", num_per, "개 추출하여 검증 데이터를 구성합니다. ")
    
    if(distribution == CONST_uniform):
        print("<----균등하게 학습데이터를 undersampling합니다.---->")
        distribution_ops = minimun_amount
    else:
        distribution_ops = -1
    category_sorted_dict_train, final_data = data_builder.build(num_per, option=option, option_uniform=distribution_ops)
    
    saveToFile(category_sorted_dict_train, dir_path, "train", subfix=option + "_" + str(mplugStyle) + "_" +str(distribution))
    saveToFile(final_data, dir_path, "valid", subfix=option + "_" + str(mplugStyle) + "_" +str(distribution))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/data/EgoOrientBench/FINAL_Dataset/other_domain_dataset.json")
    parser.add_argument("--output_path", type=str, default="./Others")
    parser.add_argument("--option", type=str, default="appending") #appending
    parser.add_argument("--distribution", type=str, default="None") #uniform
    parser.add_argument("--mplugstyle", type=bool, default=False)
    args = parser.parse_args() 

    print(list_files("../"))

    with open(args.json_path) as f:
        annotation = json.load(f)
        split_data(args.output_path, annotation, args.option, args.distribution, args.mplugstyle)