import os
import sys
import json
import base64
import requests
from urllib.parse import urlparse
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../API_EVAL')
sys.path.append(model_train_dir)

from gemini import Gemini
from chatgpt import ChatGPT
from claude import Claude
from ModelWrapper import ModelWrapper

API_MAP = {
    "gemini": Gemini,
    "chatgpt": ChatGPT,
    "claude": Claude
}

class APIModelWrapper(ModelWrapper):
    API_key = None
    model = None
    
    def __init__(self, model_name, configuration, key=None, key_path=None, **kwargs):
        self.configuration = configuration
        self.model_name = model_name
        self.getKey(key, key_path)
        self.load_model(model_name)

    def getKey(self, key=None, key_path=None):
        if key is not None:
            self.API_key = key
        elif key_path is not None:
            if key_path.endswith(".json"):
                with open(key_path, "r") as f:
                    key_dict = json.load(f)
                    self.API_key = key_dict.get(self.model_name)
            else:
                with open(key_path, "r") as f:
                    self.API_key = f.read
        
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
    
    def load_model(self, model_name):
        if model_name in API_MAP.keys():
            model_class = API_MAP[model_name]
            self.model = model_class(self.API_key, self.configuration)
        else:
            raise ValueError(f"Unsupported API model: {model_name}, Supported models: {API_MAP.keys()}")

    def predict(self, prompt, image_path):
        image, extension = self.load_image(image_path)
        image_object = {"image": image, "media_type": extension, "path": image_path}
        return self.model.inference(prompt, image_object)

if __name__ == "__main__":
    configuration = {
        "temperature": 0.2,
        "max_new_tokens": 150,
        "model_version": "gpt-4o-2024-08-06"
    }
    key_path = os.path.expanduser("~/SECRET/secrete.json")
    image_path = os.path.expanduser("~/EgoOrientBench/all_data/EgocentricDataset/imagenet_after/000210065.jpg")
    model_wrapper = APIModelWrapper("chatgpt", configuration, key_path=key_path)
    print(model_wrapper.predict("Describe the images", image_path))
