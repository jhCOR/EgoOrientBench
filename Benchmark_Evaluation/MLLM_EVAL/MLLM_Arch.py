from abc import ABC, abstractmethod
import requests
from PIL import Image
from io import BytesIO

class MLLM_Arch(ABC):
    def __init__(self, setting=None):
        self.setting = setting or self.get_default_setting()

    @abstractmethod
    def import_library(self):
        """필수 라이브러리를 가져오는 메서드"""
        pass

    @abstractmethod
    def load_model(self, model_path):
        """모델을 로드하는 메서드"""
        pass

    @abstractmethod
    def get_default_setting(self):
        """기본 설정값을 반환하는 메서드"""
        pass

    @abstractmethod
    def inference(self, prompt, image_path):
        """추론 메서드"""
        pass

    def setSetting(self, **kwargs):
        """설정을 업데이트하는 메서드"""
        self.setting.update(kwargs)
    
    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
