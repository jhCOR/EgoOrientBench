import argparse
import json
import os

from source.template import getTemplate
from source.inference import start
from source.file import count_files, checkOrMakePath, getOutputFilePath

def checkDataSetState(image_dir, json_path):
    if(os.path.exists(json_path) is True):
        print("Find the annotation data")
    else:
        print("Can't find the annotation data")
    if(os.path.exists(image_dir) is True):
        total_num = count_files(image_dir)
        print(f"Find {total_num} images in {image_dir}")
    else:
        print("Can't find the image data")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="../all_data/Image_data/imagenet_after/")
    parser.add_argument("--annotation_path", type=str, default="../all_data/Other_json/imagenet_clean_exclude_ambiguous.json")
    parser.add_argument("--output_root", type=str, default="./result")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--debug", type=str, default="False")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--mplugstyle", type=str, default="False")
    args = parser.parse_args()
    
    # 저장할 디렉터리 생성(try1, try2... 순서대로 결과물 저장)
    outputdir_path = getOutputFilePath(args.output_root, args.output_dir)
    args.output_dir = outputdir_path
    if(args.debug == "False"):
        checkOrMakePath(outputdir_path)
    
    #데이터가 경로에 정상적으로 있는지 확인
    checkDataSetState(args.image_dir, args.annotation_path)

    with open(outputdir_path+"/template.json", 'w') as f:
        json.dump(getTemplate(-1, ""), f, indent=4)
    
    # 인자 변환
    if args.end_index == -1:
        last_index = None
    else:
        last_index = args.end_index
    if args.mplugstyle == "True":
        mplug_style = True
    else:
        mplug_style = False
    
    experiment_setup = {        
        "json_path":args.annotation_path,
        "output_dir_path":outputdir_path,
        "img_dir_path":args.image_dir,

        "data_range":[args.start_index, last_index],
        "mplugStyle":mplug_style,

        "distribution":"None",
        "option":"appending"
    }
    print("[setting]: ", experiment_setup)
    start(experiment_setup)