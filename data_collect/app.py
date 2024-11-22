import os
import re

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import random
import json
import time
import argparse
import base64
from datetime import datetime

from config import Config
from source.calculate import countCategoryNum, check_duplicate_item, get_category_index
from source.data_info import list_files_in_directory, get_category_per_image_count, set_category_per_image_count
from source.file import move_file, zip_directory, remove_file
from source.data_io import get_sorted_annotation, get_annotation_data, set_annotation_data, add_data_to_json

app = Flask(__name__)

config = Config()
PREFIX = config.PREFIX
after_path = config.after_path

@app.route('/images/<path:filename>')
def serve_image(filename):
    base_path = os.path.abspath(os.path.dirname(__file__))
    image_directory = os.path.join(base_path, PREFIX)
    return send_from_directory(image_directory, filename)

@app.route('/')
@app.route('/home')
def home():
    start = time.time()
    mode_category= request.args.get('category')

    category_name_map = config.category_map
    
    result = get_category_per_image_count()
    if(mode_category is None):
        mode_category = config.category_list[0]
    path = getOneImg(PREFIX + mode_category)
 
    if not os.path.isfile(path):
        print("No File")
    
    end = time.time()
    print(f"{end - start:.5f} sec")

    img_url = f"/images/{mode_category}/{path.split('/')[-1]}"
    
    return render_template('data_labeling.html', img_path=img_url, categories=config.category_list, result=result, cate=mode_category, cate_name=category_name_map)

@app.route('/gallery')
def gallery():
    start = time.time()

    page_num = request.args.get('page')
    current_category = request.args.get('category')

    result = get_category_per_image_count()
    categories = config.category_list

    sorted_json_data = get_annotation_data()

    page_num = 0 if page_num is None else int(page_num)

    if (current_category is None) | (current_category == "None"):
        pages = int(len(sorted_json_data) / 100) + 1
        sorted_json_data = sorted_json_data[100*(page_num):100*(page_num+1)]
        selected = sorted_json_data
    
    if (current_category is not None) & (current_category != "None"):
        category_per_count = get_category_per_image_count()
        index = get_category_index(current_category, config.PREFIX)
        quick_select = sorted_json_data[index:index+category_per_count[current_category]]
        selected = quick_select
        pages = int(len(selected) / 100) + 1
        selected = selected[100*(page_num):100*(page_num+1)]

    for item in selected:
        file_name = item['path'].split("/")[-1]
        item['current_path'] = config.after_path + file_name
    
    end = time.time()
    print(f"{end - start:.5f} sec")

    return render_template('gallery.html', result=result, photos=selected, page_amount = pages, current_page = page_num, categories = categories, current = current_category)

@app.route('/search')
def search():
    name = request.args.get('target')

    if name is None:   
        return render_template('search.html', path=None)
    else:
        file_name_list = name.split(",")
        file_name_list = [after_path + name for name in file_name_list]

        return render_template('search.html', paths=file_name_list)

@app.route('/upload_to_plot', methods=['POST'])
def upload_to_plot():
    file = request.files['file-upload']

    print(f"Received file: {file.filename}")
    return 'File received successfully!'

@app.route('/cancel', methods=['POST'])
def cancel():
    if 'cancel_annotation' in request.form:
        path = request.form['path']
        current = request.form['current']
        page = request.form['page']
        name = path.split("/")[-1]

        json_data = get_annotation_data()
        check_result = check_duplicate_item(json_data, path) 
        
        if check_result is not None:
            json_data.remove(check_result)
            
            set_annotation_data(json_data)
            set_category_per_image_count(check_result['cateogry'], decrease=True)
            move_file(after_path+name, path)
            if config.additional_after_path is not None:
                remove_file(config.additional_after_path+name)

        categories = config.category_list

        if current not in categories:
            current = None
        return redirect(url_for('gallery', category=current, page=page))
    else:
        return redirect(url_for('gallery'))
    
@app.route('/button_click', methods=['POST'])
def button_click():
    if 'button_click' in request.form:
        path = request.form['path']
        answer = request.form['answer']
        category = request.form['category']

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d")

        data = {
            'path': path.replace('/images/', PREFIX), 
            'answer': answer, "cateogry":category, 
            "direction":config.dict_table.get(int(answer)), 
            "timestamp":formatted_time
        }
        print("data['path']:", data['path'])
        if len(answer) > 0:

            add_data_to_json(data)
            set_category_per_image_count(category, increase=True)
            move_file(data['path'] , after_path)

            return redirect(url_for('home', category=category))
        else: 
            return redirect(url_for('home'))
    else:
        return '오류: 버튼 클릭을 감지하지 못했습니다.'
    
@app.route('/submit_answer', methods=['POST'])
def get_answer():
    if 'button_click' in request.form:
        path = request.form['path']
        answer = request.form['answer']
        area = request.form['region']
        category = request.form['category']
        image_file = request.form['image']

        data = {
            'area': area, 'path': path, 
            'answer': answer, "cateogry":category, 
            "direction":answer
        }

        if len(answer) > 0:
            label = config.dict_table.get(answer)

            add_data_to_json(data)
            set_category_per_image_count(category, increase=True)
            move_file(path , after_path)

            if config.additional_after_path is not None:
                imgstr = re.search(r'base64,(.*)', image_file).group(1)
                img_data = base64.b64decode(imgstr)
                file_name =  path.split("/")[-1]
                with open(config.additional_after_path + file_name, 'wb') as f:
                    f.write(img_data)

            return redirect(url_for('home', category=category))
        else: 
            return redirect(url_for('home'))
    else:
        return '오류: 버튼 클릭을 감지하지 못했습니다.'

def getOneImg(dir_path):
    directory_path = dir_path
    files = list_files_in_directory(directory_path)
    
    if len(files)>0:
        random_index = random.randrange(0,len(files))
        file_path = files[random_index]
        return directory_path + "/" + file_path
    else:
        return None

def backup():
    print("백업 중...")
    zip_directory("./static", "../../../../../data/backup.zip")
    print("백업 완료...")
    return None

def make_dir(folder_path):
    if folder_path is None:
        return
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def move_already_labeled_data():

    with open(config.json_file, 'r') as f:
        annotation = json.load(f)
    test_list = []
    for row in annotation:
        path = config.PREFIX + row['path'].split("/")[-2] + "/" + row['path'].split("/")[-1]
        test_list.append(path)
        move_file(path, config.after_path)
    print(test_list[:10])

def init(strong_init):
    print("시작 중...")
    memory_path = config.memory_path

    if os.path.exists(config.json_file) is False:
        with open(config.json_file, 'w') as f:
            json.dump([], f, indent=4)

    result = countCategoryNum(PREFIX)
    with open(memory_path, 'w') as f:
        json.dump(result, f, indent=4)

    data = get_sorted_annotation()
    set_annotation_data(data)

    make_dir(config.after_path)
    if config.additional_after_path is not None:
        make_dir(config.additional_after_path)

    if strong_init is True:
        move_already_labeled_data()
    
    print("시작 세팅 완료...")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=str, default="False")
    parser.add_argument("--initialize", type=str, default="False")
    args = parser.parse_args()

    if args.backup == "True":
        backup()
    else:
        print("백업을 생략합니다.")

    strong_init = False
    if args.initialize == "True":
        strong_init = True

    init(strong_init)

    app.run(host='0.0.0.0', port=8889)