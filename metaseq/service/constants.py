# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = int(os.environ.get("INFERENCE_DEFAULT_PORT", 46010))
MODEL_PARALLEL = int(os.environ.get("INFERENCE_MODEL_PARALLEL", 16))
TOTAL_WORLD_SIZE = int(os.environ.get("INFERENCE_TOTAL_WORLD_SIZE", 16))
MAX_BEAM = int(os.environ.get("INFERENCE_MAX_BEAM", 16))

INFERENCE_CHECKPOINT_FOLDER = os.environ.get("INFERENCE_CHECKPOINT_FOLDER", "/mnt/input_data_dir/pretrained_models/OPT/175b-resharded-inference-16x1")
DEPENDENCY_FOLDER = os.environ.get("DEPENDENCY_FOLDER", "/mnt/input_data_dir/pretrained_models/OPT/dependencies")
INFERENCE_NUM_SHARDS = int(os.environ.get("INFERENCE_NUM_SHARDS", 1))
IS_FSDP_SHARED_CHECKPOINT = os.environ.get("IS_FSDP_SHARED_CHECKPOINT", "False").lower() == "true"

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import CHECKPOINT_FOLDER
except ImportError:
    # CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the
    # checkpoints from S3 are stored. As an example:
    # CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    CHECKPOINT_FOLDER = INFERENCE_CHECKPOINT_FOLDER

# tokenizer files
BPE_MERGES = os.path.join(DEPENDENCY_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(DEPENDENCY_FOLDER, "gpt2-vocab.json")
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, "reshard.pt")

LAUNCH_ARGS = [
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {MODEL_FILE}",
    "--beam 1",
    # "--distributed-port 13000",
    f"--checkpoint-shard-count {INFERENCE_NUM_SHARDS}",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]

if IS_FSDP_SHARED_CHECKPOINT:
    # If using FSDP shards, replace ddp-backend and add use-sharded-state
    LAUNCH_ARGS = ["--ddp-backend fully_sharded", "--use-sharded-state"] + LAUNCH_ARGS
else:
    LAUNCH_ARGS = ["--ddp-backend pytorch_ddp"] + LAUNCH_ARGS

# Optional arg overrides which influence model loading during inference
INFERENCE_ARG_OVERRIDES = {}
