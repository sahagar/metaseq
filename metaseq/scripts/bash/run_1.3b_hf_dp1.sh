#!/bin/bash
set -ex

root_dir=/data/users/sahagar/OPT
model_dir=/data/users/sahagar/OPT/finetuned_checkpoints
checkpoint_name=finetune_1_3b_model_1gpu_hf
chkp_dir=$model_dir/$checkpoint_name

cd $root_dir
(sudo rmdir --ignore-fail-on-non-empty $chkp_dir && rm -r $chkp_dir) | echo "Deleted..."
mkdir -p $chkp_dir

TASK_NAME=e2e_nlg
MODEL_SIZE=1.3b
MODEL_PATH=facebook/opt-$MODEL_SIZE
TOKENIZER_PATH=facebook/opt-$MODEL_SIZE
# metric
METRICS="bleu"

export CUDA_VISIBLE_DEVICES="0"

cd $root_dir/NLGPipeline/src/ych_pipeline/
python mt_r_opt_train.py \
    --data_dir="./tokenized_data" \
    --train_datasets=$TASK_NAME \
    --data_version="nomask" \
    --task_weights="1" \
    --eval_datasets=$TASK_NAME \
    --eval_data_version="mask" \
    --model_path=$MODEL_PATH \
    --tokenizer_path=$TOKENIZER_PATH \
    --device="cuda:0" \
    --learning_rate=1e-5 \
    --max_clip_norm=1 \
    --num_epochs=5 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --lr_scheduler_type="constant" \
    --model_save_dir=$chkp_dir \
    --eval_step="epoch" \
    --warmup_steps=1200 \
    --save_best \
    --metrics=$METRICS \
    --epoch_shuffle \
    --wandb_project "ABCD" \
    --skip_generation 2>&1 | tee $chkp_dir/train.log
