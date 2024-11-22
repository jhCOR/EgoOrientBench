import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Assign categories to items")
    parser.add_argument("--input_file", type=str, default="../all_data/Other_json/llava.json")
    parser.add_argument("--output_file", type=str, default="../all_data/Other_json/mplugowl.json")
    
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        items = json.load(f)

    for item in items:
        item['conversations'][0].get("value").replace("<image>", "<|image|>")

    with open(args.output_file, 'w') as f:
        json.dump(items, f, indent=4)
    
    print(f"Processed {len(items)} items and saved to {args.output_file}")

if __name__ == '__main__':
    print("데이터 후처리를 시작합니다.")
    main()
    print("데이터 후처리를 완료하였습니다.")