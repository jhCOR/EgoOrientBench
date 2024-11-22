export CUDA_VISIBLE_DEVICES=0 # You need to use GPU for excute LLaVA 13B
PREPROCESSED_FILE="../all_data/Other_json/generated_preprocessed_data_filtered.json"
python main.py --mplugstyle "False" --start_index 0 --annotation_path $PREPROCESSED_FILE

# nohup bash build_data_unify.sh > build_train_data.log 2>&1 &