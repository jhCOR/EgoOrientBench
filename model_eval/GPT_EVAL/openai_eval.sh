#!/bin/bash

python ./freeform_gpt_eval.py --mode test

# nohup bash ./openai_eval.sh > ./LOG/gpt_eval.log 2>&1 &