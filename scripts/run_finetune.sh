#!/bin/bash
gpu_ids=(0 1 2 3 4 5 6 7)
names=(
Savename
)

datasets=(
    Instruments
)

GPUS_PER_TASK=2

NUM_GPUS=${#gpu_ids[@]}
NUM_TASKS=${#names[@]}
random_numbers=($(shuf -i 2000-60000 -n $NUM_TASKS))


gpu_pool=()
for ((i=0; i<NUM_GPUS; i++)); do
    gpu_pool+=($i)
done

declare -A pid2gpus

get_free_gpus() {
    local free_gpus=("${gpu_pool[@]}")
    if [ ${#free_gpus[@]} -ge $GPUS_PER_TASK ]; then
        echo "${free_gpus[@]:0:$GPUS_PER_TASK}"
        return 0
    else
        return 1
    fi
}

reap_finished_jobs() {
    for pid in "${!pid2gpus[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            gpu_pool+=(${pid2gpus[$pid]})
            unset pid2gpus[$pid]
            gpu_pool=($(printf "%s\n" "${gpu_pool[@]}" | sort -n | uniq))
        fi
    done
}

for i in "${!names[@]}"
do
    while true; do
        reap_finished_jobs
        free_gpus_idx=($(get_free_gpus))
        if [ $? -eq 0 ]; then

            for idx in "${free_gpus_idx[@]}"; do
                for j in "${!gpu_pool[@]}"; do
                    if [[ "${gpu_pool[j]}" == "$idx" ]]; then
                        unset 'gpu_pool[j]'
                    fi
                done
            done
            gpu_pool=("${gpu_pool[@]}")

            assigned_gpu_ids=()
            for idx in "${free_gpus_idx[@]}"; do
                assigned_gpu_ids+=("${gpu_ids[$idx]}")
            done
            GPU_IDS=$(IFS=,; echo "${assigned_gpu_ids[*]}")
            break
        else
            sleep 5
        fi
    done

    echo "[INFO] Task $i | Dataset: ${datasets[$i]} | Name: ${names[$i]} | Using GPU(s): $GPU_IDS"

    mkdir -p "train_logs/${datasets[$i]}/${names[$i]}-full"
    echo "train_logs/${datasets[$i]}/${names[$i]}-full/job_${i}.log"
    echo "CUDA_VISIBLE_DEVICES=$GPU_IDS, bash scripts/finetune_full.sh ${random_numbers[$i]} ${names[$i]} ${datasets[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    bash scripts/finetune_full.sh "${random_numbers[$i]}" "${names[$i]}" "${datasets[$i]}" \
    > "train_logs/${datasets[$i]}/${names[$i]}-full/job_${i}.log" 2>&1 &

    pid=$!
    pid2gpus[$pid]="${free_gpus_idx[*]}"
done

wait
echo "All jobs finished."