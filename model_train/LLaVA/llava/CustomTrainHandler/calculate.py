def getNextTrialNum(_dir_path, splitby="_"):
    from llava.CustomTrainHandler.file import list_directories
    
    previous_trial = list_directories(_dir_path)
    if(len(previous_trial)!=0):
        prev_num = []
        
        for _name in previous_trial:
            index = int(_name.split(splitby)[-1])
            prev_num.append(index)

        prev_num.sort()

        if is_sequential_increment(prev_num) is False:
            print("폴더 인덱스가 올바르지 않습니다.")
        next_num = prev_num[-1]+1
    else:
        next_num = 1
    return next_num

def is_sequential_increment(prev_num):
    for i in range(1, len(prev_num)):
        if prev_num[i] != prev_num[i - 1] + 1:
            return False
    return True