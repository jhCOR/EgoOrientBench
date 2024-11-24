# How to Run EgoOrientBench

## 1. Installation

Download the required packages to run the model:

- **[LLaVA Repository](https://github.com/haotian-liu/LLaVA)**
- **[mPLUG-Owl2 Repository](https://github.com/X-PLUG/mPLUG-Owl)**
- **[InternVL Repository](https://github.com/OpenGVLab/InternVL)**

### Installing the InternVL Model
You can install the InternVL model using the following method:
```
cd model_train/InternVL_4B
conda env create -f environment.yaml
conda activate internvl
```

## 2. Execute Benchmark Pipline
```
cd Benchmark_Evaluation
bash scripts/run_benchmark.sh
```

Opensource MLLM Command:
```
python evaluate.py --model_type opensource --model_name llava --model_path "liuhaotian/llava-v1.5-7b"

python evaluate.py --model_type opensource --model_name internvl --checkpoint "/data/DAL_storage/pretrained/InternVL2-4B"

python evaluate.py --model_type opensource --model_name mplugowl --model_path 'MAGAer13/mplug-owl2-llama2-7b'
```

API Command:
```
python evaluate.py --model_type api --model_name chatgpt --key_path ../../SECRET/secrete.json

python evaluate.py --model_type api --model_name gemini --key_path ../../SECRET/secrete.json

python evaluate.py --model_type api --model_name claude --key_path ../../SECRET/secrete.json
```

## Full Example

[InternVL]
```
cd model_train/InternVL_4B
conda env create -f environment.yaml
conda activate internvl

cd Benchmark_Evaluation
python evaluate.py --model_type opensource --model_name internvl --checkpoint "/data/DAL_storage/pretrained/InternVL2-4B"
```