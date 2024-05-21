from loguru import logger 

from yacs.config import CfgNode as CN

config = CN()
config.DESCRIPTION = 'This the description of the configuration defaults. If you have some information related to the configuration file, please fullfil this item'


# configure the system related matters, such as gpus, cudnn and so on
config.SYSTEM = CN()
config.SYSTEM.local_rank = 0
config.SYSTEM.gpu_ids = [0]
config.SYSTEM.exp_id = "" # generated using hashlib
config.SYSTEM.final_ckpt_dir = "ckpts"
config.SYSTEM.checkpoint_dir = ""  # where to save the training log and models

# configure the things related to the data
config.DATA = CN()
config.DATA.stage = "chairs"  # name of stage is the same as README: ['chairs', 'things', 'sintel', 'kitti', 'ss']. 'ss' is self-supervised stage.
config.DATA.image_size = [384, 512] # the input size of images, height x width

# configure the training process
#-----------------basic-----------------
config.TRAIN = CN()
config.TRAIN.batch_size = 16
config.TRAIN.lr = 4e-4
config.TRAIN.num_workers = 4
config.TRAIN.weight_decay = 1e-4
config.TRAIN.grad_clip = 1.0
config.TRAIN.num_steps = 100000
config.TRAIN.seed = 326
config.TRAIN.max_flow = 400

# frequency to save some information
config.TRAIN.print_freq = 200
config.TRAIN.summary_freq = 200
config.TRAIN.save_ckpt_freq = 10000
config.TRAIN.save_latest_ckpt_freq = 1000
config.TRAIN.strategy = "C+T+S+K+H"


# configure the resume, resume pretrained model or resume training
config.RESUME = CN()
config.RESUME.use = False
config.RESUME.checkpoint = ""  # load checkpoint
config.RESUME.strict_resume = False
config.RESUME.no_resume_optimizer = False

# configure the loss function
config.TRAIN.LOSS = CN()
config.TRAIN.LOSS.L1_LOSS = CN()
config.TRAIN.LOSS.L1_LOSS.gamma = 0.9
config.TRAIN.LOSS.L1_LOSS.max_flow = 400

config.TRAIN.LOSS.VGG_LOSS = CN()
config.TRAIN.LOSS.VGG_LOSS.use = False
config.TRAIN.LOSS.VGG_LOSS.perceptual_loss_weight = [10, 10, 10, 10, 10]
config.TRAIN.LOSS.VGG_LOSS.local_vgg19_weight = "ckpts/vgg19-dcbb9e9d.pth"

# configure the model settings
#-----------------basic-----------------
config.MODEL = CN()
config.MODEL.num_scales = 1  # DO NOT CHANGE IT. output feature map scale (1/4 or 1/8 resolution)
config.MODEL.feature_channels = 128
config.MODEL.upsample_factor = 8
config.MODEL.num_transformer_layers = 6
config.MODEL.num_head = 1  # head of transformer block, must be 1 when attention_type is 'swin'
config.MODEL.attention_type = "swin"
config.MODEL.ffn_dim_expansion = 4  # feed forward network in transformer block
config.MODEL.attn_splits_list = [2]  # number of splits in swin attention
config.MODEL.prop_radius_list = [-1]  # flow propagation radius, default(-1) is global propagation, otherwise only local propagation
config.MODEL.padding_factor = 16


config.MODEL.use_k_matches = False
config.MODEL.num_k = 8


config.MODEL.GIM = CN()
config.MODEL.GIM.use_gim = False
config.MODEL.GIM.use_gim_weight = False  # if use pretrained GIM weight
config.MODEL.GIM.pretrained_gim_weight = "ckpts/loftr_outdoor.ckpt"

config.MODEL.WARPNET = CN()
config.MODEL.WARPNET.use = False
config.MODEL.WARPNET.load_gt_occlusion = False
config.MODEL.WARPNET.warp_img = False


#---------------Validation configure---------------
config.VALIDATE = CN()
config.VALIDATE.val_dataset = ['chairs']
config.VALIDATE.val_freq = 10000
config.VALIDATE.with_speed_metric = False
config.VALIDATE.eval = False
config.VALIDATE.evaluate_matched_unmatched = False  # for sintel evaluation, `matched` calculates EPE only in non-occluded regions and `unmatched` calculates EPE only in occluded regions.

#---------------Submission configure---------------
config.SUBMISSION = CN()
config.SUBMISSION.submission = False
config.SUBMISSION.output_path = "output"
config.SUBMISSION.no_save_flo = True  # if save flow as .flo file




def _get_cfg_defaults():
    """
    Get the default config template.
    NOT USE IN OTHER FILES!!
    """
    return config.clone()


def update_config(yaml_path, opts):
    """
    Make the template update based on the yaml file.
    """
    cfg = _get_cfg_defaults()
    if yaml_path != '':
        logger.info(f'Merge the config with {yaml_path}')
        cfg.merge_from_file(yaml_path)
    logger.info(f'Merge the config with the commands: {opts}')
    cfg.merge_from_list(opts)
    cfg.freeze()

    return cfg

def get_config(yaml_path):
    """
    Get the default configuration, not get the opts
    """
    cfg = _get_cfg_defaults()
    cfg.merge_from_file(yaml_path)
    cfg.freeze()

    return cfg

def get_default_yaml_file(path='./configs'):
    """
    Get the default yaml file
    Args:
        path: the location stores the default yaml
    """
    from contextlib import redirect_stdout
    import os
    cfg = _get_cfg_defaults()
    with open(os.path.join(path, 'default.yaml'), 'w') as f:
        with redirect_stdout(f):
            print(cfg.dump())
            # print('Create the default yaml')

if __name__ == '__main__':
    get_default_yaml_file()