import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../model_train/mPLUG-Owl/mPLUG-Owl2')
sys.path.append(model_train_dir)

from MLLM_Arch import MLLM_Arch

class mPLUGOwl2(MLLM_Arch):
    def __init__(self, model_setting, setting):
        self.import_library()

        self.model_setting = self.get_model_setting(model_setting)
        self.setting = self.get_default_setting(setting)

        self.model, self.tokenizer, self.processor = self.load_model()

    def import_library(self):
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates

        self.get_model_name_from_path = get_model_name_from_path
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.conv_templates = conv_templates

    def get_model_setting(self, new_model_setting):
        model_setting = {
            "model_path": 'MAGAer13/mplug-owl2-llama2-7b',
            "model_base": None,
        }
        if new_model_setting:
            model_setting.update(new_model_setting)
        return model_setting

    def get_default_setting(self, new_setting): 
        setting = {
            'conv_mode': None, 
            'sep': ',', 
            'temperature': 0.2, 
            'top_p': None, 
            'num_beams': 1, 
            'max_new_tokens': 256,
        }
        if new_setting:
            setting.update(new_setting)
        return setting
    
    def load_model(self):
        from mplug_owl2.model.builder import load_pretrained_model

        model_name = self.get_model_name_from_path(self.model_setting['model_path'])
        tokenizer, model, image_processor, context_len = load_pretrained_model(self.model_setting['model_path'], 
                                                                               self.model_setting['model_base'], model_name, 
                                                                               load_8bit=self.model_setting.get('load_8bit'), 
                                                                               load_4bit=self.model_setting.get('load_4bit'), 
                                                                               device="cuda")
        return model, tokenizer, image_processor
    
    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)

            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))

            out.append(image)
        return out
    
    def inference(self, text_input, image_path):
        conv = self.conv_templates["mplug_owl2"].copy()

        images = self.load_images([image_path])
        image_tensor = self.process_images(images, self.processor )
        image_tensor = image_tensor.to(self.model.device, dtype=torch.bfloat16)
        self.model.to(torch.bfloat16)

        inp = self.DEFAULT_IMAGE_TOKEN + text_input
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, 
                                               self.IMAGE_TOKEN_INDEX, 
                                               return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        self.model.eval()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=True if self.setting.get("temperature") > 0 else False,
                temperature=self.setting.get("temperature"),
                max_new_tokens=self.setting.get("max_new_tokens"),
                use_cache=False,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs