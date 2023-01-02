#!/bin/bash
set -ex

root_dir=/data/users/sahagar/OPT
model_dir=/data/users/sahagar/OPT/finetuned_checkpoints
data_dir=/data/users/sahagar/OPT/e2e_lm_dataset
checkpoint_name=finetune_1_3b_model_1gpu_metaseq
chkp_dir=$model_dir/$checkpoint_name

cd $root_dir
(sudo rmdir --ignore-fail-on-non-empty $chkp_dir && rm -r $chkp_dir) | echo "Deleted..."
mkdir -p $chkp_dir

# --restore-file /mnt/input_data_dir/pretrained_models/OPT/13b-fsdp-sharded-2x1/checkpoint_last.pt \

export CUDA_VISIBLE_DEVICES="0"
export MASTER_ADDR=localhost

cd $root_dir/metaseq
sudo pip install -e .
metaseq-train --task streaming_language_modeling $data_dir \
        --vocab-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-vocab.json \
        --merges-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-merges.txt \
        --save-dir $chkp_dir/ \
        --criterion cross_entropy \
        --seed 42 \
        --fixed-validation-seed 42 \
        --batch-size 8 \
        --batch-size-valid 8 \
        --num-workers 8 \
        --num-workers-valid 1 \
        --validate-interval-updates 3000 \
        --arch transformer_lm_gpt \
        --share-decoder-input-output-embed \
        --max-epoch 5 \
        --max-update 12000 \
        --dropout 0.1 \
        --optimizer adam \
        --weight-decay 0.01 \
        --clip-norm 0.0 \
        --lr 1e-5 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 1200 \
        --warmup-init-lr 1e-07 \
        --tokens-per-sample 128 \
        --sample-break-mode none \
        --fp16 \
        --decoder-layers 24 \
        --decoder-embed-dim 2048 \
        --decoder-ffn-embed-dim 8192 \
        --decoder-attention-heads 32 \
        --decoder-learned-pos \
        --log-format json 2>&1 | tee $chkp_dir/train.log