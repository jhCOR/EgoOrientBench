# 🧭 EgoOrientBench

[[Paper]](https://arxiv.org/abs/2411.16761v1)
[[How to Use]](https://github.com/jhCOR/EgoOrientBench/blob/main/How_to_USE.md)
[[Project Page]](https://jhcor.github.io/egoorientbench_is_right_right/)

## 📰 News
[2/27] Our Paper is accepted by [CVPR25](https://cvpr.thecvf.com/)! 🎉

## Is MLLM truly understanding orientation?
![image](https://github.com/user-attachments/assets/1557af5e-a946-4737-b3f1-eedc78fe2c95)

EgoOrientBench is a comprehensive framework for evaluating and improving multimodal language models (MLLMs) with a focus on enhancing and evaluating understanding of orientation.

## How to Test with Our Benchmark

Detailed instructions on how to use our benchmark can be found in the following guide:  
[**How to Use**](https://github.com/jhCOR/EgoOrientBench/blob/main/How_to_USE.md)

## General Performance Result
| Backbone      | Method    | MME Total | MMStar (↑) (NeurIPS24') | MMMU (↑) (CVPR24') | POPE (↑) (EMNLP'23) |
|--------------|----------|-----------|------------------------|--------------------|--------------------|
| LLaVA 1.5    | Zero-shot | 1792.8    | 34.67                  | 35.11              | 82.03              |
|              | Ours      | 1752.8    | 35.87 (+3.5%)          | 34.44 (-1.9%)      | 88.36 (+7.7%)      |
| mPLUG-Owl2   | Zero-shot | 1706.3    | 34.33                  | 37.55              | 86.16              |
|              | Ours      | 1727.3    | 35.27 (+2.7%)          | 38.55 (+2.7%)      | 85.60 (-0.6%)      |
| InternVL2-4B | Zero-shot | 2088.7    | 54.26                  | 47.22              | 85.91              |
|              | Ours      | 2045.9    | 53.13 (-2.1%)          | 48.00 (+1.7%)      | 85.56 (-0.4%)      |

* ↑ indicates evaluation using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

## Data

### Our Benchmark Dataset

- **JSON File**: Already available in the `all_data` folder.
- **Image Data**: Download using the command below:

#### Command to Download Image Data:
```bash
gdown https://drive.google.com/uc?id=1ZXejrBfj6E3qtHYbrUxbnqdk16_osyjI
```

[Link]: [Google Drive Link](https://drive.google.com/file/d/1ZXejrBfj6E3qtHYbrUxbnqdk16_osyjI/view?usp=drive_link)
### You need to place the image_after folder under the path EgoOrientBench/all_data/EgocentricDataset.

---

## Related Project/Dataset

- Preposition Dataset: https://github.com/amitakamath/whatsup_vlms
- MME Benchmark: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models
- VLMEvalKit: https://github.com/open-compass/VLMEvalKit
