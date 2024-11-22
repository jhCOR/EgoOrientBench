def check_image_key(one_item):
    find_a_key = None
    for test_key in ["path", "image", "image_path"]:
        if( test_key in one_item.keys() ):
            find_a_key = test_key
    return find_a_key

def extractImageList(file_json_list):
    key = check_image_key(file_json_list[0])
    image_list = []

    sorted_dict = dict(sorted(file_json_list.items(), key=lambda item: item[1]))
    for item in sorted_dict:
        image_list.append(item[key])
    