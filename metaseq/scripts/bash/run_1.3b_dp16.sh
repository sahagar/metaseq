#!/bin/bash
set -ex

root_dir=/data/users/sahagar/OPT
model_dir=/data/users/sahagar/OPT/finetuned_checkpoints
data_dir=/data/users/sahagar/OPT/e2e_lm_dataset
checkpoint_name=finetune_1_3b_model_16gpu_metaseq
chkp_dir=$model_dir/$checkpoint_name

cd $root_dir
(sudo rmdir --ignore-fail-on-non-empty $chkp_dir && rm -r $chkp_dir) | echo "Deleted..."
mkdir -p $chkp_dir

# --restore-file /mnt/input_data_dir/pretrained_models/OPT/13b-fsdp-sharded-2x1/checkpoint_last.pt \

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export MASTER_ADDR="10.217.90.175"
export WORLD_SIZE=16

cd $root_dir/metaseq
sudo pip install -e .
torchrun --standalone --nnodes=1 --nproc_per_node=16 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=29510 metaseq/cli/train.py --task streaming_language_modeling $data_dir \
        --vocab-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-vocab.json \
        --merges-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-merges.txt \
        --save-dir $chkp_dir/ \
        --criterion cross_entropy \
        --seed 42 \
        --fixed-validation-seed 42 \
        --batch-size 4 \
        --batch-size-valid 4 \
        --num-workers 0 \
        --num-workers-valid 0 \
        --validate-interval-updates 500 \
        --arch transformer_lm_gpt \
        --share-decoder-input-output-embed \
        --max-epoch 5 \
        --max-update 1500 \
        --dropout 0.1 \
        --optimizer adam \
        --weight-decay 0.01 \
        --clip-norm 0.0 \
        --lr 1e-5 \
        --lr-scheduler inverse_sqrt \
        --warmup-updates 100 \
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