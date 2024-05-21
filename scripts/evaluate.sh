#!/bin/bash
# evalute Sintel and KITTI training set
python main.py --exp_config configs/sintel.yaml --opts DATA.stage sintel MODEL.GIM.use_gim True RESUME.checkpoint ckpts/ours_sintel.pth VALIDATE.eval True
python main.py --exp_config configs/kitti.yaml --opts DATA.stage kitti MODEL.GIM.use_gim True RESUME.checkpoint ckpts/ours_kitti.pth VALIDATE.eval True
