from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from kornia.core import Module, Tensor
from kornia.feature.loftr.utils.position_encoding import PositionEncodingSine
from model.resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4, ResNetFPN_8_2_GIM

urls: dict[str, str] = {}
urls["outdoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
urls["indoor_new"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor_ds_new.ckpt"
urls["indoor"] = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_indoor.ckpt"

# Comments: the config below is the one corresponding to the pretrained models
# Some do not change there anything, unless you want to retrain it.

default_cfg = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {'initial_dim': 128, 'block_dims': [128, 196, 256]},
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross', 'self', 'cross', 'self', 'cross', 'self', 'cross'],
        'attention': 'linear',
        'temp_bug_fix': False,
    },
    'match_coarse': {
        'thr': 0.2,
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': True,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
    },
    'fine': {'d_model': 128, 'd_ffn': 128, 'nhead': 8, 'layer_names': ['self', 'cross'], 'attention': 'linear'},
}

def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

class LoFTR(Module):
    r"""Module, which finds correspondences between two images.

    This is based on the original code from paper "LoFTR: Detector-Free Local
    Feature Matching with Transformers". See :cite:`LoFTR2021` for more details.

    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.

    Args:
        config: Dict with initiliazation parameters. Do not pass it, unless you know what you are doing`.
        pretrained: Download and set pretrained weights to the model. Options: 'outdoor', 'indoor'.
                    'outdoor' is trained on the MegaDepth dataset and 'indoor'
                    on the ScanNet.

    Returns:
        Dictionary with image correspondences and confidence scores.

    Example:
        >>> img1 = torch.rand(1, 1, 320, 200)
        >>> img2 = torch.rand(1, 1, 128, 128)
        >>> input = {"image0": img1, "image1": img2}
        >>> loftr = LoFTR('outdoor')
        >>> out = loftr(input)
    """

    def __init__(self, pretrained: str | None = 'outdoor', config: dict[str, Any] = default_cfg) -> None:
        super().__init__()
        # Misc
        self.config = config
        self.initial_dim = config['resnetfpn']['initial_dim']
        
        if pretrained == 'indoor_new':
            self.config['coarse']['temp_bug_fix'] = True

        # Modules
        if config['resolution'] == (8, 2) and pretrained == 'others':
            self.backbone = ResNetFPN_8_2_GIM(config['resnetfpn'])
        elif config['resolution'] == (8, 2):
            self.backbone = ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            self.backbone = ResNetFPN_16_4(config['resnetfpn'])
        else:
            raise ValueError(f'Invalid resolution: {config["resolution"]}')

        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'], temp_bug_fix=config['coarse']['temp_bug_fix']
        )
        self.pretrained = pretrained
        

    def load_state_dict(self, state_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        named_models = [name for name, p in self.named_parameters()]
        if self.pretrained == 'others':
            state_dict = state_dict['model']
            fnt_dict = [_ for _ in state_dict.keys() if 'fnet.backbone' in _]
            for k in fnt_dict:
                if k.startswith('module.fnet.'):
                    rk = k.replace('module.fnet.', '', 1)
                    if rk in named_models:    
                        state_dict[rk] = state_dict.pop(k)
                    else:
                        state_dict.pop(k)
                else:
                    if k in named_models:
                        state_dict[k] = state_dict.pop(k)
                    else:
                        state_dict.pop(k)
            for k in list(state_dict.keys()):
                if 'backbone.' in k:
                    continue
                else:
                    state_dict.pop(k)
        else:    
            for k in list(state_dict.keys()):
                if k.startswith('matcher.'):
                    rk = k.replace('matcher.', '', 1)
                    if rk in named_models:    
                        state_dict[rk] = state_dict.pop(k)
                    else:
                        state_dict.pop(k)
                else:
                    if k in named_models:
                        state_dict[k] = state_dict.pop(k)
                    else:
                        state_dict.pop(k)
            
        # reconstruct the first layer to accept 3 channels (RGB)
        if not self.pretrained == 'none':
            torch_init_model(self, state_dict, key='model')
        self.backbone.reset_model()
        return self
   

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Args:
            data: dictionary containing the input data in the following format:

        Keyword Args:
            image0: left image with shape :math:`(N, 1, H1, W1)`.
            image1: right image with shape :math:`(N, 1, H2, W2)`.
            mask0 (optional): left image mask. '0' indicates a padded position :math:`(N, H1, W1)`.
            mask1 (optional): right image mask. '0' indicates a padded position :math:`(N, H2, W2)`.

        Returns:
            - ``keypoints0``, matching keypoints from image0 :math:`(NC, 2)`.
            - ``keypoints1``, matching keypoints from image1 :math:`(NC, 2)`.
            - ``confidence``, confidence score [0, 1] :math:`(NC)`.
            - ``batch_indexes``, batch indexes for the keypoints and lafs :math:`(NC)`.
        """
        # 1. Local Feature CNN
        _data: dict[str, Tensor | int | torch.Size] = {
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:],
            'hw1_i': data['image1'].shape[2:],
        }

        if _data['hw0_i'] == _data['hw1_i']:  # faster & better BN convergence
            feats = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            feat_c0, feat_c1 = feats.split(_data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_c1 = self.backbone(data['image0']), self.backbone(data['image1'])

        _data.update(
            {
                'hw0_c': feat_c0.shape[2:],
                'hw1_c': feat_c1.shape[2:]
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        # feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        # feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')
        feat_c0 = self.pos_encoding(feat_c0).permute(0, 2, 3, 1)
        n, h, w, c = feat_c0.shape
        feat_c0 = feat_c0.reshape(n, -1, c)

        feat_c1 = self.pos_encoding(feat_c1).permute(0, 2, 3, 1)
        n1, h1, w1, c1 = feat_c1.shape
        feat_c1 = feat_c1.reshape(n1, -1, c1)

        feat_c0 = feat_c0.permute(0, 2, 1).view(n, c, h, w)
        feat_c1 = feat_c1.permute(0, 2, 1).view(n1, c1, h1, w1)

        return feat_c0, feat_c1

    def reset_model(self):
        self.backbone.reset_model()
        return self
    
    def reset_model_resnet(self):
        self.backbone.reset_model_resnet()
        return self