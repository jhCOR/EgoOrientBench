import os
import torch
import copy
import json
import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm
import random

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from source.utils import validation_check
from source.setting import getSettingObject
from source.MonitorClass import Monitor
from source.template import getTemplate
from source.parser import direction_parser
from source.split_train_valid import split_data
CONSTANT = {
    "answer_start":"[answer]:",
    "question_start":"[question]",
    "llava_TAG":"<image>\n"
}

def sentenceParser(response):
    """질문과 답변을 파싱합니다. llava의 응답은 [Question]: "+"###\n[Answer]: 이러한 형식입니다. """
    response = response.lower()
    question = response.split(CONSTANT.get("answer_start"))[0].replace(CONSTANT.get("question_start"), "")
    answer = response.split(CONSTANT.get("answer_start"))[-1]
    return question, answer

def remove_special_characters(text):
    """정규 표현식을 사용하여 물음표, 마침표, 컴마, <, >를 제외한 모든 특수기호를 제거합니다. 주의: LLaVA는 \n<image> 태그가 있어야 합니다"""
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,?<>\u3131-\u3163\uac00-\ud7a3]', '', text)
    return cleaned_text

def image_parser(_setting):
    out = _setting.get("image_file").split( _setting.get("sep") )
    return out

def load_image(image_file):
    assert os.path.exists(image_file), f"There is no image in this path:{image_file}"

    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]

def infer_conv_mode(model_name):
    """모델 이름에 따른 대화 모드를 추론합니다."""
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"
    
def eval_model(_setting, _tokenizer, _model ,_image_processor, model_name, _context_len):
    qs = _setting.get("query")
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if _model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if _model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv_mode = infer_conv_mode(model_name)
    
    if _setting.get("conv_mode") is not None and conv_mode != _setting.get("conv_mode"):
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, _setting.get("conv_mode"), _setting.get("conv_mode")
            )
        )
    else:
        _setting['conv_mode'] = conv_mode

    conv = conv_templates[_setting.get("conv_mode")].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(_setting)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        _image_processor,
        _model.config
    ).to(_model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = _model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if _setting.get("temperature") > 0 else False,
            temperature=_setting.get("temperature"),
            top_p=_setting.get("top_p"),
            num_beams=_setting.get("num_beams"),
            max_new_tokens=_setting.get("max_new_tokens"),
            use_cache=True,
        )
    outputs = _tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def getImagePath(experiment_setup, data):
    img_path = experiment_setup.get("img_dir_path")
    input_image_path = f"{img_path}" + data['path'].split("/")[-1]

    assert os.path.exists(input_image_path), f"no exist path:{input_image_path}"
    return input_image_path

def getConvData(init, response):
    
    question_sen, answer_sen = sentenceParser(response)
    
    if init is True:
        system_prompt = '''As you are directional aware LVLM and observer that have no linkage of the object, closely take a look and answer the question'''
        question_sen = system_prompt + "\n" + CONSTANT.get("llava_TAG") + question_sen

    human_conv = {
        "from":"human",
        "value": remove_special_characters( question_sen )
    }
    gpt_conv = {
        "from":"gpt",
        "value":answer_sen
    }
    
    return human_conv, gpt_conv

def build_dataset(_annotation, experiment_setup):
    #초기화
    monitor_class = Monitor(len(_annotation))
    base_setting = getSettingObject(-1)
    print("base_setting: ", base_setting)
    
    disable_torch_init()
    model_name = get_model_name_from_path(base_setting.get("model_path"))
    print("모델: ", model_name)
    model_name = "models--liuhaotian--llava-v1.5-13b"
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
       base_setting.get("model_path"), base_setting.get("model_base"), model_name
    )
    
    # 데이터 생성 시작
    response_list = []
    for data in tqdm(_annotation):
        data['conversations'] = [] 
        data['path']="./imagenet_after/" + data['path'].split("/")[-1]
        data['image'] = data['path'].split("/")[-1]
        data['id'] = data['image'].split(".")[0]

        input_image_path = getImagePath(experiment_setup, data)
        
        monitor_class.updateCount()
        monitor_class.updateDirPerCategory(data['direction'])
        
        verbose_direction = random.choice( direction_parser.get(data['direction']) ) 
        context = f"\n[Context]: main object category : {data['category_name']}, the orientation : {verbose_direction}"
        if(data['category_name'] == None):
            print("[Caution]: ", data['id'].split("_")[0], "is no in dictionary")

        data_collection = {
            "conv"    :copy.deepcopy(data), 
            "detail"  :copy.deepcopy(data), 
            "complex" :copy.deepcopy(data)
            }
        
        for i in [1]:
            # 각 index에 따라 설정이 다르므로 대응되는 설정 객체를 가져옵니다. 
            case_setting = getSettingObject(i)
            prompt = getTemplate(i,context)

            case_setting['query'] = prompt
            case_setting['image_file'] = input_image_path
            
            response = eval_model(case_setting, tokenizer, model, image_processor, model_name, context_len)
            human_conv, gpt_conv = getConvData( ( i in [0, 1, 2]), response)

            if(i == 0):
                data = data_collection['conv']
                data['type'] = "conv"
            elif(i == 1):
                data = data_collection['complex']
                data['type'] = "complex"
            elif(i == 2):
                data = data_collection['detail']
                data['type'] = "detail"

            data['conversations'].append(human_conv)
            data['conversations'].append(gpt_conv)
            
            monitor_class.updataPrompt(prompt, data['path'], response)

        for key in data_collection.keys():
            response_list.append( data_collection.get(key) )

        dir_path = experiment_setup.get("output_dir_path")
        save_path = f"{dir_path}/imagenet_LLaVA_13B_"+str(experiment_setup.get("data_range"))+"_"+".json"
        with open(save_path, 'w') as f:
            json.dump(response_list, f, indent=4)
    return response_list, monitor_class.dir_per_category

def start(experiment_setup):
    json_path = experiment_setup.get("json_path")

    annotation = None
    with open(json_path, 'r') as f:
        annotation = json.load(f)
        print("전체 모 데이터 집합 개수: ", len(annotation))
        if(experiment_setup.get("data_range")[1] is None):
            print("SHUFFLE")
            random.shuffle(annotation)
        annotation = annotation[experiment_setup.get("data_range")[0]: experiment_setup.get("data_range")[1]]

        print("총 개수: ", len(annotation))

    annotation, trash_can = validation_check(annotation)

    category_exist = {}
    for row in annotation:
        if(category_exist.get(row['cateogry']) is None):
            category_exist[row['cateogry']] = 0
        category_exist[row['cateogry']] = category_exist[row['cateogry']] + 1
    print("총 카테고리 개수: ", len(list(category_exist.keys())))

    final_build_dataset, category_distribution = build_dataset( annotation, experiment_setup)
    
    print("검증데이터 분할")
    split_data(experiment_setup.get("output_dir_path"), final_build_dataset,
               experiment_setup.get("option"), experiment_setup.get("distribution"),
                experiment_setup.get("mplugStyle") )
    sorted_by_value  = dict(sorted(category_distribution.items(), key=lambda item: item[1]))
    print(sorted_by_value)
    print(final_build_dataset[:5])
    print("Work Finish")