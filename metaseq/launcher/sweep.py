# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import sys
import os
import subprocess
import itertools
import random
import hashlib
import logging
import time
from pathlib import Path
from collections import OrderedDict
from typing import Optional, List, Callable, MutableMapping
from urllib.parse import urlparse

import metaseq
from metaseq.utils import get_random_port

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger(__name__)

class hyperparam(object):
    """Base class for defining hyperparameters."""

    def __init__(
        self,
        name,
        values=None,
        binary_flag=False,
        save_dir_key=None,
        positional_arg=False,
    ):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-tensorboard`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        - positional_arg : whether the hyperparameter is a positional argument
        """
        self.name = name
        if values is None:  # syntactic sugar for binary flags
            self.values = [True]
            self.binary_flag = True
        else:
            self.values = values if isinstance(values, list) else [values]
            self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key
        self.positional_arg = positional_arg
        self.current_value = None

        if positional_arg and name.startswith("-"):
            raise ValueError(
                f"positional arguments must not start with a dash ({name})"
            )

        if len(self.values) > 1 and self.save_dir_key is None:
            raise ValueError(
                f"{name} has more than one value but is missing a save_dir_key!"
            )

    def get_cli_args(self):
        if self.binary_flag:
            return [self.name] if self.current_value else []
        elif self.positional_arg:
            return [self.current_value]
        else:
            return [self.name, self.current_value]

    def get_save_dir_key(self):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if self.current_value else None
        return self.save_dir_key(self.current_value)


def _get_args(add_extra_options_func=None, input_args: Optional[List[str]] = None):
    """
    input_args (List[str]): strings to parse, defaults to sys.argv
    """
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps ")
    parser.add_argument("--grid", help="grid function we used", default=None)

    parser.add_argument("-d", "--data", help="path to data directory")
    parser.add_argument("--valid-subsets", nargs='*', default=[])
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-t",
        "--num-trials",
        default=-1,
        type=int,
        help="number of random hyperparam configurations to try (-1 for grid search)",
    )
    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        default=0,
        help="update freq",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="resume any runs that failed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument("--local", action="store_true", help="run job locally")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--script", default="metaseq/cli/train.py", help="script to launch"
    )
    parser.add_argument(
        "--python", default="python", help="path to nonstandard python binary"
    )

    # Slurm params
    parser.add_argument(
        "--salloc", action="store_true", help="run agaist current allocation"
    )
    parser.add_argument("--reservation", help="reservation to run on")
    parser.add_argument(
        "--exclusive", action="store_true", help="if set, get exclusive host"
    )
    parser.add_argument(
        "--time", default="4320", help="expected job duration in minutes"
    )
    parser.add_argument("--mem", "--mem", help="memory to request")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help='gpu constraint, if any. e.g. "volta"',
    )
    parser.add_argument("--comment", help="comment string")
    parser.add_argument(
        "--snapshot-code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=".",
        help="root path for saving the snapshot code.",
    )
    parser.add_argument(
        "--snapshot-recurse-dirs-internal",
        default="metaseq_internal",
        help="comma-separated directories from where to recursively copy *.py, *.so and *.yaml files",
    )
    parser.add_argument(
        "--snapshot-recurse-dirs-oss",
        default="metaseq",
        help="comma-separated directories from where to recursively copy *.py, *.so and *.yaml files",
    )
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable tensorboard logging"
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable WandB logging")
    parser.add_argument(
        "--post-steps",
        nargs="+",
        help="additional steps to execute after the primary job is complete. "
        "this can be a file with the steps, or a string. some placeholders such as "
        "{job_dir} will be replaced",
    )

    # Env flags
    parser.add_argument("--azure", action="store_true", help="running on azure")
    parser.add_argument("--aws", action="store_true", help="running on aws")
    parser.add_argument("--fair", action="store_true", help="running on fair")
    parser.add_argument("--rsc", action="store_true", help="running on rsc")
    
    # Azure specific flag
    parser.add_argument(
        "--full-azure-upload-path",
        default=None,
        help="Azure blob storage SAS URL",
    )

    parser.add_argument(
        "--azure-folder-auto-name",
        action="store_true",
        help="Automatically name azure folder",
    )

    # Following args have env specific defaults.
    parser.add_argument(
        "--partition",
        help="slurm partition to run on",
    )
    parser.add_argument(
        "--checkpoints-dir",
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument("--cpus-per-task", type=str)
    parser.add_argument(
        "--cpu-bind", help="configured to improve all-to-all perf, especially on A100s"
    )
    parser.add_argument(
        "--local-checkpoints-dir",
        help="node-local directory for saving checkpoints",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,  # None will default to save_dir/tb
        help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "-ts",
        "--tombstonable",
        type=bool,
        default=False,
        help=(
            "make the job killable by writing a "
            "tombstone 'tombstone_<job_id>' file to user's home directory "
            "(/shared/home/$USER)"
        ),
    )

    if add_extra_options_func is not None:  # mutates parser
        add_extra_options_func(parser)
    args = parser.parse_args(input_args)

    # Set defaults based on env
    _modify_arg_defaults_based_on_env(args)
    return args


def _modify_arg_defaults_based_on_env(args):
    # TODO(susan): move all this default logic into separate config file
    default_checkpoint_dir = "checkpoints" # str(datetime.date.today())
    
    # assign default checkpoint directory
    if args.checkpoints_dir is None:
        args.checkpoints_dir = default_checkpoint_dir

    # assign default # cpus per task
    if args.cpus_per_task is None:
        args.cpus_per_task = 12

    # assign default cpu bind
    if args.cpu_bind is None:
        args.cpu_bind = (
            "mask_cpu:ffffff000000,ffffff000000,ffffff,ffffff,"
            "ffffff000000000000000000,ffffff000000000000000000,"
            "ffffff000000000000,ffffff000000000000"
        )

def set_env(args, env):
    if "NCCL_SOCKET_IFNAME" not in env:
        env["NCCL_SOCKET_IFNAME"] = "eth0"

    # NCCL_ASYNC_ERROR_HANDLING allows failfast upon NCCL error.
    # It only takes effect in torch 1.7+
    if "NCCL_ASYNC_ERROR_HANDLING" not in env:
        env["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-timeout
    if "NCCL_IB_TIMEOUT" not in env:
        env["NCCL_IB_TIMEOUT"] = "22"

    # Avoid failure "Call to ibv_reg_mr failed" for NCCL2.4.x
    if "NCCL_TREE_THRESHOLD" not in env:
        env["NCCL_TREE_THRESHOLD"] = "0"

    # Print NCCL info by default
    if "NCCL_DEBUG" not in env:
        env["NCCL_DEBUG"] = "INFO"

    # NCCL speed up default
    if "NCCL_NSOCKS_PERTHREAD" not in env:
        env["NCCL_NSOCKS_PERTHREAD"] = "4"

    if "NCCL_SOCKET_NTHREADS" not in env:
        env["NCCL_SOCKET_NTHREADS"] = "2"

    env["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(args.num_gpus)])
    env["WORLD_SIZE"] = str(args.num_nodes * args.num_gpus)

def gen_train_command(args, config, save_dir, env):
    # generate train command
    code_folder = str(Path(metaseq.__file__).parents[1])
    # train_cmd = [args.python, os.path.join(code_folder, args.script)]
    train_cmd = ["torchrun", f"--nnodes={args.num_nodes}", f"--nproc_per_node={args.num_gpus}", f"--node_rank={os.environ.get('RANK', 0)}", f"--master_addr={os.environ.get('MASTER_ADDR', 'localhost')}", "--master_port=29510", os.path.join(code_folder, args.script)]
    train_cmd.extend(["--distributed-world-size", str(args.num_nodes * args.num_gpus)])

    assert args.data is not None, "data path must be specified"
    assert save_dir is not None, "save_dir must be specified"

    train_cmd.extend([args.data])
    train_cmd.extend(["--save-dir", save_dir])
    train_cmd.extend(["--save-async"])
    
    if not args.no_wandb:
        try:
            import wandb
        except ImportError:
            wandb = None
        if wandb or ("WANDB_API_KEY" in env and "WANDB_BASE_URL" in env):
            if "--wandb-project" not in config:
                project = args.prefix
                train_cmd.extend(["--wandb-project", project])
            if "WANDB_RUN_GROUP" not in env:
                env["WANDB_RUN_GROUP"] = args.prefix
            if "WANDB_RUN_ID" not in env:
                env["WANDB_RUN_ID"] = hashlib.md5(save_dir.encode("utf-8")).hexdigest()
            if "WANDB_RESUME" not in env:
                env["WANDB_RESUME"] = "allow"

    if not args.no_tensorboard:
        if args.tensorboard_logdir is None:
            tensorboard_logdir = os.path.join(save_dir, "tb")
        else:
            tensorboard_logdir = os.path.join(
                args.tensorboard_logdir,
                args.prefix,
            )
        train_cmd.extend(["--tensorboard-logdir", tensorboard_logdir])
    
    for hp in config.values():
        train_cmd.extend(map(str, hp.get_cli_args()))
    return train_cmd

def main(
    get_grid: Callable[[argparse.Namespace], List[hyperparam]],
    postprocess_hyperparams: Callable[
        [argparse.Namespace, MutableMapping[str, hyperparam]], None
    ],
    add_extra_options_func: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    scheduler_args: Optional[List[str]] = None,
) -> None:
    """Do a grid search.

    Parameters:
        get_grid: A unary callable which returns the grid to search over.
            The callable is passed the parsed sweep arguments including the extra
            arguments defined by `add_extra_options_func`. See also `get_args`.
            The returned list represents the dimensions of the grid. That is, a list of
            length n represents a grid of dimension n. Let v_i denote the number of
            possible values for dimension i. Then the total number of configurations
            is given by v_1 * ... * v_n.
        postprocess_hyperparams: A 2-ary callable to post-process hyperparameter
            configurations before running the job. The first argument is the parsed
            sweep arguments including the extra arguments defined by
            `add_extra_options_func`. The second argument is a realized hyperparameter
            configuration as a mutable mapping of hyperparameter name to `hyperparam`
            instance with a `current_value` set.
        add_extra_options_func: A unary callable which adds extra arguments to the
            sweep CLI. It is passed the parser used to define the sweep script's CLI.
        scheduler_args: A list of unprocessed arguments to parse. If None, then
            `sys.argv[1:]`.
    """
    args = _get_args(add_extra_options_func, scheduler_args)
    from metaseq.launcher.slurm import main as backend_main

    get_grid = get_grid[args.grid] if args.grid is not None else get_grid
    
    grid = get_grid(args)
    # grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    
    # set environment
    env = os.environ.copy()
    set_env(args, env)

    save_dir = os.path.join(args.checkpoints_dir, args.prefix)
    env["METASEQ_SAVE_DIR"] = save_dir

    # if args.distributed_rank == 0:
    # create save directory if it doesn't exist
    logger.info(f"creating save directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # start training
    config = OrderedDict()
    for hp in grid:
        config[hp.name] = hp
        config[hp.name].current_value = hp.values[0]

    # postprocess hyperparams
    postprocess_hyperparams(args, config)

    # generate train command
    train_cmd = gen_train_command(
        args,
        config,
        save_dir,
        env
    )

    train_stdout = os.path.join(save_dir, "train.log")
    logger.info(f"running command: {train_cmd}")
    logger.info(f"Train Log: {train_stdout}")

    exception = None
    try:
        subprocess.run(train_cmd, check=True, env=env)
    except Exception as e:
        exception = e

    # Re-throw exception if any
    if exception:
        # Exceptions printed here may not be caught.
        # ITP searches for error pattern in last 2KB of the log. Errors from mpi
        # jobs are not caught, causing 3 retries regardless of the error type.
        # ITP is increasing log size limit to 1MB.
        logger.error(exception)
        sys.exit(1)

    # train_proc = subprocess.Popen(train_cmd, env=env)
    # train_proc.wait()

    # with subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env) as train_proc, \
    #         open(train_stdout, "w") as train_stdout_h:
    #     train_proc.wait()
    #     stdout = train_proc.stdout.read().decode("utf-8")
    #     print(stdout, file=train_stdout_h)
    #     if train_proc.returncode != 0:
    #         logger.error("train command failed. Traceback:")
    #         logger.error(stdout[stdout.rfind("Traceback"):])
    #         sys.exit(1)
