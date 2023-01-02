#!/bin/bash
set -ex

root_dir=/data/users/sahagar/OPT
model_dir=/data/users/sahagar/OPT/finetuned_checkpoints
data_dir=/data/users/sahagar/OPT/e2e_lm_dataset
checkpoint_name=finetune_13b_model_metaseq
chkp_dir=$model_dir/$checkpoint_name

cd $root_dir
(sudo rmdir --ignore-fail-on-non-empty $chkp_dir && rm -r $chkp_dir) | echo "Deleted..."
mkdir -p $chkp_dir

cd $root_dir/metaseq
sudo pip install -e .
python metaseq/launcher/opt_baselines.py --task streaming_language_modeling --num-nodes 1 --num-gpus 16 \
    --script metaseq/cli/train.py \
    --model-size 13b --model-parallel 2 \
    --checkpoints-dir $model_dir --prefix $checkpoint_name \
    --restore-file /mnt/input_data_dir/pretrained_models/OPT/13b-fsdp-sharded-2x8/checkpoint_last.pt \
    --data $data_dir \
    --vocab-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-vocab.json --merges-filename /mnt/input_data_dir/pretrained_models/OPT/dependencies/gpt2-merges.txt \
    --log-interval 25 --warmup-updates 100 --validate-interval-updates 150 \
    --batch-size 8 \
    --max-epochs 5 \
    --max-updates 1500 \
    --lr 1e-5 \
    --seq-len 128 \
    --reset-dataloader \
    --keep-last-epochs 1 2>&1 | tee $chkp_dir/train.log