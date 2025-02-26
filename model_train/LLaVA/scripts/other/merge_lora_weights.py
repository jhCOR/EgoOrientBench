import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import os

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save_model_path", type=str, required=True)
    
    args = parser.parse_args()

    if os.path.exists(args.save_model_path) is False:
        os.makedirs(args.save_model_path)
    merge_lora(args)
