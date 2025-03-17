#!/bin/bash

# args
name="finetuning_512"
config_file=configs/${name}/config.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="../../outputs"

mkdir -p $save_root/$name

# Detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# Set GPU devices - use all available GPUs
GPU_DEVICES=$(seq -s "," 0 $((NUM_GPUS-1)))
echo "Using GPUs: ${GPU_DEVICES}"

## run training
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python3 -m torch.distributed.launch \
--nproc_per_node=${NUM_GPUS} --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
../../main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices ${NUM_GPUS} \
lightning.trainer.num_nodes=1