#!/bin/bash
# phase I: AIL
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_config configs/chairs.yaml --opts DATA.stage chairs
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_config configs/things.yaml --opts DATA.stage things
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_config configs/sintel.yaml --opts DATA.stage sintel
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_config configs/kitti.yaml --opts DATA.stage kitti

# phase II: KGL
CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_config configs/ss.yaml --opts DATA.stage ss
