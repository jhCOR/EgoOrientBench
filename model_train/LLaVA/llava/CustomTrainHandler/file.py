import os
from llava.CustomTrainHandler.calculate import getNextTrialNum
def list_directories(path):
    try:
        # 해당 경로의 파일과 디렉터리 목록을 불러옵니다
        items = os.listdir(path)
        directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
        
        return directories
    except FileNotFoundError:
        return f"The directory {path} does not exist."
    except PermissionError:
        return f"Permission denied to access {path}."
    except Exception as e:
        return f"An error occurred: {e}"

def count_files(path):
    try:
        # 해당 경로의 파일과 디렉터리 목록을 불러옵니다
        items = os.listdir(path)
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        return len(files)
    except FileNotFoundError:
        return f"The directory {path} does not exist."
    except PermissionError:
        return f"Permission denied to access {path}."
    except Exception as e:
        return f"An error occurred: {e}"
    
def checkOrMakePath(_path):
    try:
        if(os.path.exists(_path) is False):
            os.mkdir(_path)
        return True
    except Exception as e:
        print("checkOrMakePath Error: ", e)
        return False
    
#[Notice] Code Below is Not Universal Code
def getOutputFilePath(_dir, _name, prefix="try_"):
    current_output_path = None
    if(len(_name)<1):
        current_trial_num = getNextTrialNum(_dir)
        current_output_file_name = prefix + str(current_trial_num)
    else:
        current_output_file_name = _name
    current_output_path = _dir + "/" + current_output_file_name

    return current_output_path