
begin_cross_layer=$1
save_name=$2
Datasets=$3
text_contrast_weight=$4
image_contrast_weight=$5
recon_contrast_weight=$6
OUTPUT_DIR=log/$Datasets/${save_name}
mkdir -p $OUTPUT_DIR

python -u main.py \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.0 \
  --device cuda:0 \
  --text_data_path ../data/$Datasets/$Datasets.emb-llama-td.npy \
  --image_data_path ../data/$Datasets/$Datasets.emb-ViT-L-14.npy \
  --ckpt_dir $OUTPUT_DIR \
  --eval_step 2 \
  --batch_size 2048 \
  --begin_cross_layer $begin_cross_layer \
  --use_cross_rq True \
  --text_class_info ../data/$Datasets/$Datasets.index_lemb_kmeans512.json \
  --image_class_info ../data/$Datasets/$Datasets.index_vitemb_kmeans512.json \
  --text_contrast_weight $text_contrast_weight \
  --image_contrast_weight $image_contrast_weight \
  --recon_contrast_weight $recon_contrast_weight \
  --epochs 1000 > $OUTPUT_DIR/train.log
