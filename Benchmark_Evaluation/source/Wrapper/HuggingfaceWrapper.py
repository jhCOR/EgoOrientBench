from ModelWrapper import ModelWrapper
from transformers import AutoModel, AutoTokenizer

import os
import sys
import json
import base64
import requests
from urllib.parse import urlparse
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../Huggingface_EVAL')
sys.path.append(model_train_dir)

from qwenvl import QwenVL

Huggingface_MAP = {
    "qwenvl": QwenVL,
}

class HuggingfaceModelWrapper(ModelWrapper):
    model = None

    def __init__(self, model_name, configuration, **kwargs):
        self.configuration = configuration
        self.model_name = model_name
        self.load_model(model_name)

    def load_model(self, model_name):
        if model_name in Huggingface_MAP.keys():
            model_class = Huggingface_MAP[model_name]
            self.model = model_class(self.configuration)
        else:
            raise ValueError(f"Unsupported API model: {model_name}, Supported models: {Huggingface_MAP.keys()}")
        
    def load_image(self, image_path):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            # URL 경로인 경우
            response = requests.get(image_path)
            if response.status_code == 200:
                # URL에서 확장자 추출
                parsed_url = urlparse(image_path)
                extension = os.path.splitext(parsed_url.path)[-1].lower()
                if not extension:
                    raise ValueError(f"Could not determine file extension from URL: {image_path}")
                return base64.b64encode(response.content).decode('utf-8'), extension
            else:
                raise ValueError(f"Failed to fetch image from URL: {image_path}, status code: {response.status_code}")
        elif os.path.exists(image_path):
            extension = os.path.splitext(image_path)[-1].lower()
            # 로컬 파일 경로인 경우
            if self.model_name == "gemini":
                image = Image.open(image_path)
                return image, extension
            with open(image_path, "rb") as image_file:
                # 로컬 파일에서 확장자 추출
                if not extension:
                    raise ValueError(f"Could not determine file extension from path: {image_path}")
                return base64.b64encode(image_file.read()).decode('utf-8'), extension
        else:
            path = os.path.abspath(image_path)
            print(f"Invalid image path: {path}")
            raise ValueError(f"Invalid image path: {image_path}")
        
    def predict(self, prompt, image_path, load_image_file=False):
        if load_image_file:
            image, extension = self.load_image(image_path)
        else:
            image = None
            extension = None
        image_object = {"image": image, "media_type": extension, "path": image_path}
        return self.model.inference(prompt, image_object)

if __name__ == "__main__":

    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = HuggingfaceModelWrapper("qwenvl", None)
    print(model_wrapper.predict("Describe the images", image_path, load_image_file=False))