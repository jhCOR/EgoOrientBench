import os

def getNextTrialNum(_dir_path, splitby="_"):
    
    previous_trial = list_directories(_dir_path)
    if(len(previous_trial)!=0):
        prev_num = []
        
        for _name in previous_trial:
            index = int(_name.split(splitby)[-1])
            prev_num.append(index)

        prev_num.sort()

        if is_sequential_increment(prev_num) is False:
            print("[Warning] The folder index is not sequential.")
        next_num = prev_num[-1]+1
    else:
        next_num = 1
    return next_num

def list_directories(path):
    try:
        items = os.listdir(path)
        directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
        
        return directories
    except FileNotFoundError:
        return f"The directory {path} does not exist."
    except PermissionError:
        return f"Permission denied to access {path}."
    except Exception as e:
        return f"An error occurred: {e}"

def is_sequential_increment(prev_num):
    for i in range(1, len(prev_num)):
        if prev_num[i] != prev_num[i - 1] + 1:
            return False
    return True