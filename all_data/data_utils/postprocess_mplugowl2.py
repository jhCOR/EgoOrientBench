import argparse
import json
import os

def main():    
    parser = argparse.ArgumentParser(description="Assign categories to items")
    parser.add_argument("--input_file" , type=str, default=os.path.expanduser("~/EgoOrientBench/all_data/mPLUG-Owl2_json/imagenet_QA_7b_clean_test.json"))
    parser.add_argument("--output_file", type=str, default=os.path.expanduser("~/EgoOrientBench/all_data/mPLUG-Owl2_json/imagenet_QA_7b_clean_test.json_mplugowl"))
    
    args = parser.parse_args()

    # 입력 파일 불러오기
    with open(args.input_file, 'r') as f:
        items = json.load(f)

    # 각 항목에 카테고리 이름을 할당
    
    for item in items:
        item['conversations'][0]['value'] = item['conversations'][0].get("value").replace("<image>", "<|image|>")
    
    # 출력 파일에 저장
    with open(args.output_file, 'w') as f:
        json.dump(items, f, indent=4)
    
    print(f"Processed {len(items)} items and saved to {args.output_file}")

if __name__ == '__main__':
    print("데이터 후처리를 시작합니다.")
    main()
    print("데이터 후처리를 완료하였습니다.")