if __name__ == "__main__":
    from llava.train.train import train
    from llava.CustomTrainHandler.CustomTrainHandler import getOutputDirPath
    
    automatic_setting = {
        "result_output_dir": getOutputDirPath("./MyResult")
    }
    print("\nautomatic_setting: ", automatic_setting)
    train(automatic_setting, attn_implementation="flash_attention_2")