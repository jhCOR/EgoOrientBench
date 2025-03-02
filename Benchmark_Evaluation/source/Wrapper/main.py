import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../source/Wrapper')
sys.path.append(model_train_dir)

from HuggingfaceWrapper import HuggingfaceModelWrapper
from APIWrapper import APIModelWrapper
from OpensourceWrapper import OpensourceModelWrapper

from supported_models import SUPPORTED_MODEL_MAP

MODEL_WRAPPER_MAP = {
    "huggingface": HuggingfaceModelWrapper,
    "api": APIModelWrapper,
    "opensource": OpensourceModelWrapper,
}

def create_model_wrapper(model_type, model_name, configuration, custom_image_loder=None, **kwargs):
    supported_model_list = SUPPORTED_MODEL_MAP.get(model_type, [])
    print("supported_model_list: ", supported_model_list)
    
    if model_name not in supported_model_list:
        raise ValueError(f"Unsupported model type or name: {model_type} {model_name}")

    model_wrapper_class = MODEL_WRAPPER_MAP.get(model_type)
    if not model_wrapper_class:
        raise ValueError(f"No wrapper class found for model type: {model_type}")
    
    return model_wrapper_class(model_name, configuration, **kwargs)

def testAPIWrapper():
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
        "model_version": "gpt-4o-2024-08-06"
    }

    key_path = os.path.expanduser("~/SECRET/secrete.json")
    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = create_model_wrapper("api", "chatgpt", configuration, key_path=key_path)
    print(model_wrapper.predict("Describe the images", image_path))

def testHuggingfaceWrapper():

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = create_model_wrapper("huggingface", "qwenvl", None)
    print(model_wrapper.predict("Describe the images", image_path))

def testOpensourceWrapper():
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
    }
    model_setting = {
        "checkpoint": '/data/DAL_storage/pretrained/InternVL2-4B'
    }

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = create_model_wrapper("opensource", "internvl", configuration, model_setting=model_setting)
    print(model_wrapper.predict("Describe the images", image_path))

if __name__ == "__main__":
    # testAPIWrapper()
    # testOpensourceWrapper()
    # testHuggingfaceWrapper()
    pass
