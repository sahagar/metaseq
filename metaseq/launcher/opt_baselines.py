#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.
"""
import os

from metaseq.launcher.opt_job_constants import (
    TOTAL_TRAIN_TOKENS,
    TOTAL_WARMUP_TOKENS,
    MODEL_SIZES,
    VALID_SUBSETS,
)
from metaseq.launcher.sweep import (
    hyperparam, main as sweep_main,
)

import logging
import sys
import time

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger(__name__)

def add_extra_options_func(parser):
    # NOTE we shouldn't add new options here... track changes via git instead
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--reset-dataloader",
        action="store_true",
        help="reset the dataloader to epoch 1",
    )
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), required=True)
    
    # Args related to benchmarking and profiling
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use synthetic data and only train for 50 steps (for benchmarking)",
    )
    parser.add_argument(
        "--profile",
        default=False,
        action="store_true",
    )
    parser.add_argument("--max-updates", "--mu", type=int, default=None)
    parser.add_argument("--warmup-updates", type=int, default=None)
    parser.add_argument("--max-epochs", "--me", type=int, default=None)
    parser.add_argument(
        "--disable-validation", action="store_true", help="skip doing validation"
    )
    
    parser.add_argument("--save-interval-updates", type=int, default=2000)
    parser.add_argument("--validate-interval-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--model-parallel", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--task", type=str, default="streaming_language_modeling")
    parser.add_argument("--vocab-filename", type=str, default="gpt2-vocab.json")
    parser.add_argument("--merges-filename", type=str, default="gpt2-merges.txt")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--aim-repo", type=str, default=None)

def get_grid(args):
    # Infer data path if not given
    DATA_ROOT = ""
    if args.valid_subsets == []:
        valid_subsets = [""]
    else:    
        valid_subsets = args.valid_subsets # VALID_SUBSETS

    SEQ_LEN = 2048
    EST_SEQ_LEN = SEQ_LEN

    size = MODEL_SIZES[args.model_size]
    # updates = 300B tokens / 2048 seq_len / 1024 batchsize
    
    total_gpus = args.num_gpus * args.num_nodes

    if args.seq_len: SEQ_LEN = args.seq_len

    if args.lr:
        size.lr = args.lr
    elif args.batch_size:
        est_bz = (size.batch_size // total_gpus) // SEQ_LEN
        size.lr = size.lr * round((args.batch_size/est_bz), 6)
        
        if SEQ_LEN != EST_SEQ_LEN:
            size.lr = size.lr * round((SEQ_LEN/EST_SEQ_LEN), 6) 

    if args.model_parallel: size.model_parallel = args.model_parallel
    if args.batch_size: size.batch_size = args.batch_size

    # TODO: fix training to run with 1 gpu (see Enable sweep scripts to run with a single GPU #176)
    if args.num_gpus < 2:
        raise ValueError(f"Need at least two gpus to run model parallel code. num_gpus={args.num_gpus}")
    if total_gpus < size.model_parallel:
        raise ValueError(
            f"Total gpus (num_gpus={args.num_gpus} * num_nodes={args.num_nodes}) must be greater than model parallel factor. mp={size.model_parallel}, total_gpus={total_gpus}"
        )
    if total_gpus % size.model_parallel != 0:
        raise ValueError(
            f"Total gpus (num_gpus * num_nodes) must be divisible by model parallel factor. mp={size.model_parallel}, total_gpus={total_gpus}"
        )
    if size.n_heads % size.model_parallel != 0:
        raise ValueError(
            f"Number of heads must be divisible by model parallel factor. mp={size.model_parallel}, n_heads={size.n_heads}"
        )

    total_gpus = (args.num_gpus * args.num_nodes) // size.model_parallel
    ddp_bsz = size.batch_size # (size.batch_size // total_gpus) // SEQ_LEN
    total_updates = args.max_updates
    total_epochs = args.max_epochs
    if total_updates is None:
        total_updates = int(TOTAL_TRAIN_TOKENS)
    # warmup_updates = int(TOTAL_WARMUP_TOKENS) // size.batch_size
    warmup_updates = args.warmup_updates
    log_interval = args.log_interval

    grid = []

    # default streaming_lm task config
    task_config = [
        hyperparam("--task", args.task),
        hyperparam("--sample-break-mode", "none"),
        hyperparam("--vocab-filename", args.vocab_filename),
        hyperparam("--merges-filename", args.merges_filename),
    ]
    
    # separate task config for dummy_lm
    if args.benchmark:
        # Overrides for speed benchmarking
        task_config = [
            hyperparam("--task", "dummy_lm"),
            hyperparam("--dict-size", 51200 - 4) # TODO(susan): what is this -4 sorcery? relic of more nmt things?
        ]
    
        args.save_interval_epochs = 0
        args.save_interval_updates = 0

        total_updates = 50
        warmup_updates = 50
        log_interval = 5

    grid += task_config

    if args.profile:
        grid += [hyperparam("--profile")]

    grid += [
        hyperparam(
            "--valid-subset", ",".join(f"valid/{ss}" for ss in valid_subsets)
        ),
        hyperparam("--save-interval-updates", args.save_interval_updates),
        hyperparam("--train-subset", "train"),
        hyperparam("--ignore-unused-valid-subsets"),
        hyperparam("--num-workers", 8),
        hyperparam("--num-workers-valid", 1),
        hyperparam("--validate-interval-updates", args.validate_interval_updates),
        hyperparam("--memory-efficient-fp16"),
        hyperparam("--fp16-init-scale", 4),
        
        # we set this for the main run but it's probably nt needed here
        # hyperparam("--threshold-loss-scale", 0.25),
        
        hyperparam("--ddp-backend", "fully_sharded"),
        hyperparam("--use-sharded-state"),
        # ZeRO-2
        hyperparam("--no-reshard-after-forward"),
        
        hyperparam("--checkpoint-activations"),
        hyperparam("--model-parallel-size", size.model_parallel),
        hyperparam("--criterion", "vocab_parallel_cross_entropy"),
        hyperparam("--distribute-checkpointed-activations"),
        hyperparam("--tensor-parallel-init-model-on-gpu"),
        # Flags to match exact same initialization of Megatron code for exp 12.00
        hyperparam("--full-megatron-init"),
        hyperparam("--megatron-init-sigma", 0.006),
        hyperparam("--activation-fn", "relu"),
        hyperparam("--arch", "transformer_lm_megatron"),
        hyperparam("--share-decoder-input-output-embed"),
        
        hyperparam("--decoder-layers", size.n_layers),
        hyperparam("--decoder-embed-dim", size.emb_size),
        hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
        hyperparam("--decoder-attention-heads", size.n_heads),
        
        # Switch to learned position embeddings for exp 12.00, without scaling
        hyperparam("--decoder-learned-pos"),
        hyperparam("--no-scale-embedding"),
        hyperparam("--tokens-per-sample", SEQ_LEN),
        hyperparam("--optimizer", "adam"),
        
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam("--adam-betas", f"(0.9, 0.95)"),
        
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam("--adam-eps", 1e-8),
        
        # GPT-3 used --clip-norm=1.0
        hyperparam("--clip-norm", 1.0),
        
        hyperparam("--clip-norm-type", "l2"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", size.lr),
        hyperparam("--end-learning-rate", size.lr * 0.1),
        hyperparam("--warmup-updates", warmup_updates),
        hyperparam("--total-num-update", total_updates),
        hyperparam("--dropout", 0.1),
        hyperparam("--attention-dropout", 0.1),
        hyperparam("--no-emb-dropout"),
        hyperparam("--weight-decay", 0.1),
        hyperparam("--batch-size", ddp_bsz),
        hyperparam("--update-freq", 1),
        hyperparam("--max-update", total_updates),
        hyperparam("--seed", 1),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    
    if args.restore_file:
        grid += [hyperparam("--restore-file", args.restore_file)]
    if args.reset_dataloader:
        grid += [hyperparam("--reset-dataloader")]

    if args.disable_validation:
        grid += [hyperparam("--disable-validation")]

    if args.max_epochs is not None:
        grid += [hyperparam("--max-epoch", total_epochs)]
    if args.aim_repo:
        grid += [hyperparam("--aim-repo", args.aim_repo)]
        
    return grid


def postprocess_hyperparams(args, config):
    pass


def cli_main():
    sweep_main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )


if __name__ == "__main__":
    cli_main()
