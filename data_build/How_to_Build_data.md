# This Part is show how to build Data

[Training Data]



[Benchmark]
- Put your raw dataset in "EgoOrientBench/data_build/dataset/Source"
Dataset should be:
```
{
    "path": "./imagenet_after/COCO_val2014_000000358765.jpg",
    "answer": "3",
    "direction": "front right",
    "category_name": "bird",
    "domain": "real",
    "base_dataset": "D3_Eval",
    "conversations": []
}
```
- run: EgoOrientBench/data_build/build_benchmark.py

# [Notice1] 
- If you haven't run LLAVA before, uncomment the following.

```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

# [Notice2] 
- If you encounter the message:
```
 The new behaviour of LlamaTokenizer 
(with `self.legacy = False`) requires the protobuf library but it was not found in your environment.
```
# then you need to run : pip install protobuf