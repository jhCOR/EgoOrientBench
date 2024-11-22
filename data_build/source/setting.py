from datetime import datetime

config_setting = {
    "model_path": "liuhaotian/llava-v1.5-13b",
    "model_base": None,
    "query": None,
    "conv_mode": None,
    "image_file": None,
    "sep": ",",
    "TIMESTAMP": datetime.today().strftime("%Y%m%d_%H:%M:%S"),
}

simple_config_base = {
    "temperature": 0.2,
    "top_p": 0.5,
    "num_beams": 1,
    "max_new_tokens": 1024,
}

detail_config_base = {
    "temperature": 0.5,
    "top_p": 0.5,
    "num_beams": 1,
    "max_new_tokens": 512,
}

complex_config_base = {
    "temperature": 0.5,
    "top_p": 0.5,
    "num_beams": 1,
    "max_new_tokens": 512,
}
# 기본 설정값을 복사하고 새로운 설정값으로 업데이트(1)
simple_config = config_setting.copy()
simple_config.update(simple_config_base)

# 기본 설정값을 복사하고 새로운 설정값으로 업데이트(2)
detail_config = config_setting.copy()
detail_config.update(detail_config_base)

# 기본 설정값을 복사하고 새로운 설정값으로 업데이트(3)
complex_config = config_setting.copy()
complex_config.update(complex_config_base)

def getSettingObject(index):
    config_setting_list = [simple_config, detail_config, complex_config]
    if(index == -1):
        return config_setting
    else:
        if(index == 0):
            return config_setting_list[0]
        elif(index == 1):
            return config_setting_list[1]
        elif(index == 2):
            return config_setting_list[2]
