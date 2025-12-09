#!/bin/bash

name=$1
Dataset=$2
OUTPUT_DIR=../data/$Dataset

for model_name in $name; do
  echo "Processing $model_name"
  python -u generate_indices_distance.py \
    --dataset $Dataset \
    --text_data_path ../data/$Dataset/$Dataset.emb-llama-td.npy \
    --image_data_path ../data/$Dataset/$Dataset.emb-ViT-L-14.npy \
    --device cuda:0 \
    --ckpt_path log/$Dataset/$model_name/best_text_collision_model.pth \
    --output_dir $OUTPUT_DIR \
    --output_file ${Dataset}.index_lemb_${model_name}.json \
    --content text
  python -u generate_indices_distance.py \
      --dataset $Dataset \
      --text_data_path ../data/$Dataset/$Dataset.emb-llama-td.npy \
      --image_data_path ../data/$Dataset/$Dataset.emb-ViT-L-14.npy \
      --device cuda:0 \
      --ckpt_path log/$Dataset/$model_name/best_image_collision_model.pth \
      --output_dir $OUTPUT_DIR \
      --output_file ${Dataset}.index_vitemb_${model_name}.json \
      --content image
done

