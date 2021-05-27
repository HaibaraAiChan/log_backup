#!/bin/bash
export RANK=0
export WORLD_SIZE=1

CUDA_VISIBLE_DEVICES=0 deepspeed --num_gpus=1  train_serial.py --deepspeed_config ds_config.json