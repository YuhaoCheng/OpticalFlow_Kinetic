import torch


import time
import argparse
import numpy as np
import os
from glob import glob
from model.model import KinetickOpticalFlow
from utils import misc
from utils.utils import create_exp_dir, gen_md5
from evaluate import evaluate, submission
from train import train
from loguru import logger
from torchinfo import summary


from utils.config import update_config


def get_args_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--exp_config', type=str, default='configs/default.yaml', help='yaml configuration files')  # reading parameters from yaml configuration file
    
    parser.add_argument('--opts', help='change the config from the command-line', default=None, nargs=argparse.REMAINDER)
    return parser

def main(config):
    if config.SYSTEM.local_rank == 0:
        misc.save_args(config)
        misc.check_path(config.SYSTEM.checkpoint_dir)
        misc.save_command(config.SYSTEM.checkpoint_dir)

    seed = config.TRAIN.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = KinetickOpticalFlow(config,
                                feature_channels=config.MODEL.feature_channels,
                                num_scales=config.MODEL.num_scales,
                                upsample_factor=config.MODEL.upsample_factor,
                                num_head=config.MODEL.num_head,
                                attention_type=config.MODEL.attention_type,
                                ffn_dim_expansion=config.MODEL.ffn_dim_expansion,
                                num_transformer_layers=config.MODEL.num_transformer_layers,
                                ).to(device)
    model = torch.nn.DataParallel(model)
    if config.SYSTEM.local_rank == 0 and not config.VALIDATE.eval and not config.SUBMISSION.submission:
        logger.info(f'Model summary:\n{summary(model, verbose=0)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.lr,
                                  weight_decay=config.TRAIN.weight_decay)
    


    start_epoch = 0
    start_step = 0

    # resume checkpoints
    if config.RESUME.use:
        logger.info(f'Load checkpoint: {config.RESUME.checkpoint}')

        loc = 'cuda:{}'.format(config.SYSTEM.local_rank)
        checkpoint = torch.load(config.RESUME.checkpoint, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model.load_state_dict(weights, strict=config.RESUME.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                config.RESUME.no_resume_optimizer:
            logger.info('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        logger.info(f'start_epoch: {start_epoch}, start_step: {start_step}')

    # evaluate
    if config.VALIDATE.eval:
        evaluate(model, config)
        return

    # Sintel and KITTI submission
    if config.SUBMISSION.submission:
        submission(model, config)
        return
    
    train(model=model, config=config, optimizer=optimizer, start_epoch=start_epoch, start_step=start_step, device=device)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    config = update_config(args.exp_config, args.opts)

    
    time_str = time.strftime('%Y%m%d-%H%M%S')

    config.defrost()
    config.SYSTEM.exp_id = gen_md5(time_str+config.DATA.stage)
    
    if config.SYSTEM.local_rank == 0:
        if config.VALIDATE.eval:
            eval_str = 'eval_'
        else:
            eval_str = ''
        config.SYSTEM.checkpoint_dir = os.path.join('exps', f'{time_str}_k{config.MODEL.num_k}_{config.SYSTEM.exp_id}_{eval_str}KineticsOpticalFlow_{config.DATA.stage}')
        os.makedirs(config.SYSTEM.checkpoint_dir, exist_ok=True)
        
        logger.add(os.path.join(config.SYSTEM.checkpoint_dir, f'{time_str}_k{config.MODEL.num_k}_{config.SYSTEM.exp_id}_{eval_str}KineticsOpticalFlow_{config.DATA.stage}.log'))

        scripts = glob('*.py') + glob('data') + glob('model') + glob('scripts') + glob('utils') + glob('configs')
        create_exp_dir(config.SYSTEM.checkpoint_dir, scripts_to_save=scripts)
        logger.info(args)
        logger.info(config)
    config.freeze()

    main(config)
