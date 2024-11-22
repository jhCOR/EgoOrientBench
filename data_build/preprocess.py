import argparse
import json
import os

# 카테고리 맵을 불러오는 함수
def getCategoryMap(category_file):
    """카테고리 매핑 파일을 읽어 사전으로 반환"""
    category_dictionary = None
    with open(category_file, 'r') as f:
        category_dictionary = json.load(f)
    return category_dictionary

# 메인 함수
def main():
    # argparse로 입력받은 인자 처리
    parser = argparse.ArgumentParser(description="Assign categories to items")
    parser.add_argument("--input_file", type=str, default="../all_data/Other_json/annotation.json")
    parser.add_argument("--img_dir", type=str, default="../all_data/Image_data/imagenet_after/")
    parser.add_argument("--category_file", type=str, default="../all_data/Other_json/category_mapping.json")
    parser.add_argument("--output_file", type=str, default="../all_data/Other_json/generated_preprocessed_data.json")
    
    args = parser.parse_args()

    # 카테고리 맵 불러오기
    category_map = getCategoryMap(args.category_file)

    # 입력 파일 불러오기
    with open(args.input_file, 'r') as f:
        items = json.load(f)

    items_to_remove = []
    for item in items:
        item['path'] = "./imagenet_after/" + item['path'].split("/")[-1]
        if os.path.exists(args.img_dir + item['path'].split("/")[-1]) is False:
            print("이미지가 존재하지 않음: ", item['path'])
            items_to_remove.append(item)  # 삭제할 항목을 리스트에 추가
            continue  # 삭제할 항목은 이후 작업을 건너뜀
        item['conversations'] = []
        category_id = item.get('cateogry')
        if category_id in category_map:
            item['category_name'] = category_map[category_id]

        else:
            print("카테고리 매칭 오류", item.get("path"))
            item['category_name'] = "Unknown"
    
    print(len(items_to_remove))
    for item in items_to_remove:
        print(item['path'])
    # 출력 파일에 저장
    with open(args.output_file, 'w') as f:
        json.dump(items, f, indent=4)
    
    print(f"Processed {len(items)} items and saved to {args.output_file}")

if __name__ == '__main__':
    print("데이터 전처리를 시작합니다.")
    main()
    print("데이터 전처리를 완료하였습니다.")