#!/bin/bash

exe_py_path=/root/anaconda3/envs/CRN/bin/python

SCRIPT_PATH=$(dirname "$(realpath "$0")")
echo "$SCRIPT_PATH"
cd $SCRIPT_PATH

# $exe_py_path -m torch.distributed.launch --nproc_per_node 8 train_radar_5ch.py \
#     --cfg 'models/yolov5s_5ch.yaml' \
#     --data 'data/train_0712_5ch.yaml' \
#     --hyp 'data/hyps/obb/hyp.finetune_dota_radar.yaml' \
#     --batch-size 512 \
#     --workers 4 \
#     --img 1024 \
#     --epochs 200 \
#     --device 0,1,2,3,4,5,6,7 \
#     --save-period 2 \
#     --sync-bn


# $exe_py_path exps/det/CRN_r50_256x704_128x128_4key.py --amp_backend native -b 16 --gpus 4
$exe_py_path exps/det/CRN_r50_512x768_128x128_4key.py --amp_backend native -b 16 --gpus 4