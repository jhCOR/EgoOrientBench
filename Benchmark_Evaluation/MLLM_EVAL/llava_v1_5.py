import os
import re
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
model_train_dir = os.path.join(current_dir, '../../../model_train/LLaVA')
sys.path.append(model_train_dir)

from MLLM_Arch import MLLM_Arch

class LLaVA1_5(MLLM_Arch):
    def __init__(self, model_setting, setting):
        self.import_library()

        self.model_setting = self.get_model_setting(model_setting)
        self.setting = self.get_default_setting(setting)

        self.model, self.tokenizer, self.processor = self.load_model()
    
    def import_library(self):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            get_model_name_from_path
        )
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates

        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER

        self.conv_templates = conv_templates

        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.get_model_name_from_path = get_model_name_from_path

    def get_model_setting(self, new_model_setting):
        model_setting = {
            "model_path": "liuhaotian/llava-v1.5-7b",
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
        from llava.model.builder import load_pretrained_model

        model_name = self.get_model_name_from_path(self.model_setting['model_path'])
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_setting['model_path'], 
            self.model_setting['model_base'],
            model_name
        )
        return model, tokenizer, image_processor

    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out

    def inference(self, prompt, image_path):
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN

        if self.IMAGE_PLACEHOLDER in prompt:
            if self.model.config.mm_use_im_start_end:
                prompt = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                prompt = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.model.config.mm_use_im_start_end:
                prompt = image_token_se + "\n" + prompt
            else:
                prompt = self.DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv_mode = "llava_v1"
        self.setting['conv_mode'] = conv_mode
         
        conv = self.conv_templates[self.setting.get("conv_mode")].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = self.load_images([image_path])
        image_sizes = [x.size for x in images]
        images_tensor = self.process_images(
            images,
            self.processor,
            self.model.config
        ).to(self.model.device, dtype=torch.bfloat16) 
        self.model.to(torch.bfloat16) 

        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, 
                                       self.IMAGE_TOKEN_INDEX, 
                                       return_tensors="pt").unsqueeze(0).cuda()
        )

        self.model.eval()
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.setting.get("temperature") > 0 else False,
                temperature=self.setting.get("temperature"),
                top_p=self.setting.get("top_p"),
                num_beams=self.setting.get("num_beams"),
                max_new_tokens=self.setting.get("max_new_tokens"),
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs


