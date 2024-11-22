# Installation

[1] conda env create --name internvl --file environment.yml
[2] pip install flash-attn==2.3.6 --no-build-isolation

# Train
[1] cd internvl_chat/shell/internvl2.0

# Eval

[1] cd internvl_chat/script
[2] 
```
nohup bash eval_preposition.sh
nohup bash eval_MME.sh
bash tuning_eval.sh
```