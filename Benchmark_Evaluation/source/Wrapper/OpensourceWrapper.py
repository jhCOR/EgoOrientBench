from ModelWrapper import ModelWrapper
import os
import sys
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../MLLM_EVAL')
sys.path.append(model_train_dir)

from supported_models import SUPPORTED_MODEL_MAP

Opensource_MAP = {
    SUPPORTED_MODEL_MAP['opensource'][0]: "llava_v1_5.LLaVA1_5",
    SUPPORTED_MODEL_MAP['opensource'][1]: "mplugowl2.mPLUGOwl2",
    SUPPORTED_MODEL_MAP['opensource'][2]: "internvl2.InternVL",
}

class OpensourceModelWrapper(ModelWrapper):

    def __init__(self, model_name, configuration,  model_setting = None, **kwargs):
        self.model_name = model_name
        self.model_setting = model_setting
        self.configuration = configuration
        self.load_model()

    def load_model(self):
        if self.model_name in Opensource_MAP.keys():
            module_class = Opensource_MAP[self.model_name]  # 예: "llava_v1_5.LLaVA1_5"
            module_name, class_name = module_class.rsplit(".", 1)  # 모듈과 클래스 이름 분리
            module = importlib.import_module(module_name)  # 모듈 동적 임포트
            model_class = getattr(module, class_name)  # 클래스 동적 가져오기
            self.model = model_class(self.model_setting, self.configuration)
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}. Supported models: {list(Opensource_MAP.keys())}")

    def predict(self, prompt, image_path):
        return self.model.inference(prompt, image_path)

def test_llava():
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
    }
    model_setting = {
        "model_path": "liuhaotian/llava-v1.5-7b",
        "model_base": None,
    }

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = OpensourceModelWrapper("llava", configuration, model_setting=model_setting)
    print(model_wrapper.predict("Describe the images", image_path))

def test_mplugowl():
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
    }
    model_setting = {
        "model_path": 'MAGAer13/mplug-owl2-llama2-7b',
        "model_base": None,
    }

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = OpensourceModelWrapper("mplugowl", configuration, model_setting=model_setting)
    print(model_wrapper.predict("Describe the images", image_path))

if __name__ == "__main__":
    # test_llava()
    # test_mplugowl()
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
    }
    model_setting = {
        "checkpoint": '/data/DAL_storage/pretrained/InternVL2-4B'
    }

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = OpensourceModelWrapper("internvl", configuration, model_setting=model_setting)
    print(model_wrapper.predict("Describe the images", image_path))