
export CUDA_VISIBLE_DEVICES=1

python clip_feature.py \
    --image_root /userhome/dataset/amazon18/Images \
    --save_root /userhome/dataset/MQL4GRec \
    --model_cache_dir /userhome/cache_models/clip \
    --dataset Instruments


