
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
port=$1
save_name=$2
Index_file=.index_lemb_$save_name.json
Image_index_file=.index_vitemb_$save_name.json
    
Tasks='seqrec,seqimage,item2image,image2item,seqimage2item,seqitem2image'
Valid_task=seqrec

Datasets=$3

load_model_name=./config/ckpt

OUTPUT_DIR=./log/$Datasets-$save_name-contrastive-align-0.01-temp-0.07-full-second
mkdir -p $OUTPUT_DIR
log_file=$OUTPUT_DIR/train.log
Per_device_batch_size=1024
torchrun --nproc_per_node=2 --master_port=$port finetune_contrastive.py \
    --data_path ./data/ \
    --dataset $Datasets \
    --output_dir $OUTPUT_DIR \
    --base_model $load_model_name \
    --per_device_batch_size $Per_device_batch_size \
    --learning_rate 1e-3 \
    --epochs 200 \
    --weight_decay 0.01 \
    --save_and_eval_strategy epoch \
    --logging_step 50 \
    --max_his_len 20 \
    --prompt_num 4 \
    --patient 10 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --tasks $Tasks \
    --valid_task $Valid_task > $log_file


results_file=$OUTPUT_DIR/results_${Valid_task}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task}_20.json

torchrun --nproc_per_node=2 --master_port=$port test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items >> $log_file

Valid_task=seqimage
results_file=$OUTPUT_DIR/results_${Valid_task}_20.json
save_file=$OUTPUT_DIR/save_${Valid_task}_20.json

torchrun --nproc_per_node=2 --master_port=$port test_ddp_save.py \
    --ckpt_path $OUTPUT_DIR \
    --data_path ./data/ \
    --dataset $Datasets \
    --test_batch_size 64 \
    --num_beams 20 \
    --index_file $Index_file \
    --image_index_file $Image_index_file \
    --test_task $Valid_task \
    --results_file $results_file \
    --save_file $save_file \
    --filter_items >> $log_file 

python ensemble.py \
    --output_dir $OUTPUT_DIR\
    --dataset $Datasets\
    --data_path ./data/\
    --index_file $Index_file\
    --image_index_file $Image_index_file\
    --num_beams 20

