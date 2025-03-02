import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

class QwenVL:
    def __init__(self, setting):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    def load_library(self):
        pass

    def setSetting(self, **kwargs):
        pass

    def inference(self, prompt, image_object):
        query = self.tokenizer.from_list_format([
            {'image': image_object['path']},
            {'text' : prompt},
        ])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response