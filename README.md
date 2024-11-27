# EgoOrientBench

[[Preprint]](https://arxiv.org/abs/2411.16761v1)
[[How to Use]](https://github.com/jhCOR/EgoOrientBench/blob/main/How_to_USE.md)

## Is MLLM truly understanding orientation?
![image](https://github.com/user-attachments/assets/1557af5e-a946-4737-b3f1-eedc78fe2c95)

EgoOrientBench is a comprehensive framework for evaluating and improving multimodal language models (MLLMs) with a focus on enhancing and evaluating understanding of orientation.

## Overview

- **Data Build**
- **Tuning**
  - LLaVA
  - InternVL
  - mPLUG-Owl2
- **Evaluation**
  - EgoOrientBench(Ours)
  - Preposition
  - MME
---

## How to Test with Our Benchmark

Detailed instructions on how to use our benchmark can be found in the following guide:  
[**How to Use**](https://github.com/jhCOR/EgoOrientBench/blob/main/How_to_USE.md)

---

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
