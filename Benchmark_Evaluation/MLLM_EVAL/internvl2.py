import os
import sys
import torch
from argparse import Namespace

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../model_train/InternVL_4B/internvl_chat')
sys.path.append(model_train_dir)

from MLLM_Arch import MLLM_Arch

class InternVL(MLLM_Arch):
    def __init__(self, model_setting, setting):
        self.import_library()

        self.model_setting = self.get_model_setting(model_setting)
        self.setting = self.get_default_setting(setting)

        self.model, self.tokenizer, self.processor = self.load_model()
    
    def import_library(self):
        from internvl.model import load_model_and_tokenizer
        from internvl.train.dataset import build_transform, dynamic_preprocess

        self.load_model_and_tokenizer = load_model_and_tokenizer
        self.build_transform = build_transform
        self.dynamic_preprocess = dynamic_preprocess

    def get_model_setting(self, new_model_setting):
        model_setting = {
            'checkpoint': "/data/pretrained/InternVL2-4B",
            'load_in_8bit': False, 
            'load_in_4bit': False,
            "auto": False,
        }
        if new_model_setting:
            model_setting.update(new_model_setting)
        return model_setting

    def get_default_setting(self, new_setting):
        setting = {
            'num_beams': 1,
            'top_k': 50, 
            'top_p': None, 
            'sample': False, 
            'dynamic': False, 
            'max_num': 6,
            "temperature": 0.2,
            "max_new_tokens": 256,
        }
        if new_setting:
            setting.update(new_setting)
        return setting 
    
    def load_images(self, image_file, input_size=224):
        try:
            image = self.load_image(image_file)
        except Exception as e:
            print(f"Failed to load image {image_file}: {e}")
            return None
        
        transform = self.build_transform(is_train=False, input_size=input_size)
        if self.setting['dynamic']:
            images = self.dynamic_preprocess(image, image_size=input_size,
                                        use_thumbnail=use_thumbnail,
                                        max_num=self.setting['max_num'])
        else:
            images = [image]
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def load_model(self):
        args = Namespace(**self.model_setting)
        model, tokenizer = self.load_model_and_tokenizer(args)
        return model, tokenizer, None

    def inference(self, prompt, image_path):
        image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        question = "<image>\n"+prompt

        if not os.path.exists(image_path):
            print(f"Warning: Image file '{image_path}' does not exist. Skipping this entry.")
            print("check: ", os.path.abspath(image_path))
            return ""

        loaded_image = self.load_images(image_path, image_size)
        if loaded_image is not None:
            pixel_values = loaded_image.cuda().to(torch.bfloat16)
        else:
            return ""

        generation_config = dict(
            num_beams= self.setting.get("num_beams"),
            max_new_tokens= self.setting.get("max_new_tokens", 256),
            do_sample= True if self.setting.get("temperature") > 0 else False,
            temperature= self.setting.get("temperature", 0.2),
        )
        response = self.model.chat(
            tokenizer= self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            verbose=True
        )

        return response