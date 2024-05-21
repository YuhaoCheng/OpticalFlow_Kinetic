import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loftr.loftr import LoFTR
from model.backbone import CNNEncoder
from model.transformer import FeatureTransformer, FeatureFlowAttention
from model.matching import compute_kmap
from model.geometry import flow_warp, coords_grid, normalize_img, feature_add_position

from torchvision import models
import numpy as np
from loguru import logger

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.  
    """
    def __init__(self, local_weight=None, requires_grad=False):
        super().__init__()
        if local_weight is not None:
            weight = torch.load(local_weight, map_location='cpu')
        else:
            weight = models.VGG19_Weights.DEFAULT
        vgg_pretrained_features = models.vgg19(weights=weight).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu5]
        return out



class ResidualBlock_motion(nn.Module):
    """
    stacks of Residual Block for MotionDecoder to obtain optical flow
    """
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super().__init__()
        self.in_planes = in_planes
        self.planes = planes
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if in_planes != planes:
            self.post_process = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.in_planes != self.planes:
            x = self.post_process(x)

        return self.relu(x+y)


class FeatureMotionGenerator(nn.Module):
    """decode optical flow from feature pairs"""
    def __init__(self, feature_dim=256, extra_dim=8, output_dim=128, norm_fn='batch', dropout=0.0):
        super().__init__()
        self.in_planes = feature_dim*2+extra_dim
        
        self.output_dim = output_dim
        self.norm_fn = norm_fn
        self.dropout = dropout
        self.layer1 = self._make_layer(dim=self.in_planes, stride=1)
        self.layer2 = self._make_layer(dim=256, stride=1)
        self.layer3 = self._make_layer(dim=self.output_dim, stride=1)
        self.flow_head = nn.Sequential(nn.Conv2d(self.output_dim//2, 128, 3, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 2, 3, padding=1)
                                       )
        

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock_motion(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock_motion(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    
    def forward(self, x):
        """predict motion from source to target"""
        # x is concat([feature0, feature1])

        _, c, h, w = x.shape  
        
        out = self.layer1(x)
        
        out = self.layer2(out)
        out = self.layer3(out)
        # split by channel
        out, _ = torch.chunk(out, chunks=2, dim=1)
        out = self.flow_head(out)

        return out
    

class WarpNet(nn.Module):
    """
    input: feature map or image and optical flows
    output: warped_feature_map or warped_img, occlusion map inferred from optical flow
    """
    def __init__(self, in_dim=128+4, out_dim=128+1):
        super().__init__()
        self.in_dim = in_dim  # 2 flows + src
        self.out_dim = out_dim # occ + warped_src
        channel = [in_dim, 128, 192, 256, 192, 128, out_dim]
        self.layer1 = nn.Conv2d(channel[0], channel[1], kernel_size=7, padding=3)
        self.layer2 = nn.Conv2d(channel[1], channel[2], kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(channel[2], channel[3], kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(channel[3], channel[4], kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(channel[4], channel[5], kernel_size=3, padding=1)
        self.layer6 = nn.Conv2d(channel[5], channel[6], kernel_size=7, padding=3)
        self.post_process = nn.Conv2d(channel[1], channel[2], kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f_flow, b_flow, source):
        # warp source based on flow to target, occlusion_prediction
        x = torch.cat([f_flow, b_flow, source], dim=1)
        out_1 = self.layer1(x)
        out = self.relu(self.layer2(out_1))
        out = self.relu(self.layer3(out+self.post_process(out_1)))
        out = self.relu(self.layer4(out))
        out = self.relu(self.layer5(out))
        out = self.relu(self.layer6(out))
        
        occ_map, warped_source = torch.split(out, [1, self.out_dim-1], dim=1)
        occ_map = torch.sigmoid(occ_map)  # 1: non-occ; 0: occ
        f_flow = f_flow * occ_map
        warped_source = flow_warp(source, f_flow)
        return warped_source, occ_map


class KinetickOpticalFlow(nn.Module):
    def __init__(self,
                 config,
                 feature_channels=128,
                 num_scales=1,
                 upsample_factor=8,
                 num_head=1,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 **kwargs,
                 ):
        super().__init__()
        self.config = config
        self.model_config = config.MODEL
        self.train_config = config.TRAIN
        self.gim_config = config.MODEL.GIM
        self.warpnet_config = config.MODEL.WARPNET
        self.loss_config = config.TRAIN.LOSS

        self.num_scales = self.model_config.num_scales
        self.feature_channels = self.model_config.feature_channels
        self.upsample_factor = self.model_config.upsample_factor
        self.attention_type = self.model_config.attention_type
        self.num_transformer_layers = self.model_config.num_transformer_layers

        # backbone: part of Apparent Information Encoder
        if self.gim_config.use_gim:
            logger.info(f'Loading loftr')
            local_weight=self.gim_config.pretrained_gim_weight
            if self.gim_config.use_gim_weight:                
                self.backbone = LoFTR(pretrained='outdoor')
                logger.info(f'Loading checkpoint: {local_weight}')
                pretrained_dict = torch.load(local_weight, map_location='cpu')
                self.backbone.load_state_dict(state_dict=pretrained_dict['state_dict'])
            else:
                self.backbone = LoFTR(pretrained='none')
                self.backbone.reset_model()  # delete non-use layers
            self.conv1x1 = nn.Conv2d(in_channels=256, out_channels=feature_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)
            
        # Transformer: part of Apparent Information Encoder
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )
        
        # Motion Decoder
        self.feature_motion_generator = FeatureMotionGenerator(feature_dim=feature_channels, extra_dim=self.model_config.num_k, output_dim=feature_channels)

        # WarpNet
        if self.warpnet_config.use:
            self.warpnet = WarpNet(in_dim=feature_channels+4, out_dim=feature_channels+1)
            if self.warpnet_config.warp_img:        
                self.convb = nn.Conv2d(in_channels=feature_channels, out_channels=3, kernel_size=1, stride=1)
        
        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
        ## for perceptual loss
        if self.loss_config.VGG_LOSS.use:
            self.vgg = Vgg19(local_weight=self.loss_config.VGG_LOSS.local_vgg19_weight)
        
        
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low

        # reverse: resolution from low to high
        features = features[::-1]

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        return feature0, feature1

    def extract_feature_loftr(self, img0, img1):
        loftr_input = {}
        loftr_input['image0'] = img0
        loftr_input['image1'] = img1
        features_img0, features_img1 = self.backbone(loftr_input)
        return features_img0, features_img1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)

            mask = self.upsampler(concat)
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            mask = torch.softmax(mask, dim=2)

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow

    # Kinetics-guided Motion Generator
    def kinetics_supervisor(self, flow, feature0, delta_t=0.5):
        teacher_flow = flow * delta_t
        # generated warped feature map based on teacher flow
        if self.warpnet_config.use:
            feature_warped, _ = self.warpnet(teacher_flow, teacher_flow, feature0)
        else:
            feature_warped = flow_warp(feature0, teacher_flow)
        aux_k, _, _, _ = compute_kmap(feature0, feature_warped, k=self.model_config.num_k)
        # aux_k: [B, num_k, H*W]
        B, _, H, W = feature0.shape
        aux_k = aux_k.view(B, -1, H, W)
        concat_f = torch.cat([feature0, feature_warped, aux_k], dim=1)

        student_flow = self.feature_motion_generator(concat_f)
        
        return student_flow

    def forward(self, img0, img1,
                attn_splits_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):
        results_dict = {}
        flow_preds = []
        if self.config.DATA.stage == 'ss':
            student_flow_list = []
            teacher_flow_list = []

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]
        
        if self.gim_config.use_gim:
            feature0, feature1 = self.extract_feature_loftr(img0, img1)  # list of features
            feature0_list = [self.conv1x1(feature0)]
            feature1_list = [self.conv1x1(feature1)]
        else:
            # resolution low to high
            feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features
        
        flow = None

        assert len(attn_splits_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]

            attn_splits = attn_splits_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)

            # Transformer: self+cross attention to augment features
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)

            # coords0, coords1 = self.initialize_flow(img0)
            # get top K matches between feature0 and feature1
            aux_k, _, _, _ = compute_kmap(feature0, feature1, k=self.model_config.num_k)
            # aux_k: [B, num_k, H*W]
            B, _, H, W = feature0.shape
            aux_k = aux_k.view(B, -1, H, W)
            # generate motion based on aux_k, as it represents the most possible flow
            # skip correlation volumn, predict motion from features instead
            concat_fmap = torch.cat([feature0, feature1, aux_k], dim=1)                

            flow_pred = self.feature_motion_generator(concat_fmap)
            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred

            # upsample to the original resolution for supervison
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)

                        
            if scale_idx == self.num_scales - 1:
                # kinetics guided learning phase                                          
                if self.config.DATA.stage == 'ss':
                    student_flow = self.kinetics_supervisor(flow, feature0)
                    
                    student_flow_list.append(student_flow)
                    teacher_flow_list.append(flow)  
                    
                if self.warpnet_config.use and self.config.DATA.stage == 'sintel_occ':
                    # generate warped_img, occ_pred    
                    
                    # predict the backward optical flow by exchanging feat1 and feat0
                    aux_k_b, _, _, _ = compute_kmap(feature1, feature0, k=self.model_config.num_k)
                    # aux_k: [B, num_k, H*W]
                    B, _, H, W = feature1.shape
                    aux_k_b = aux_k_b.view(B, -1, H, W)
                    
                    concat_fmap_b = torch.cat([feature1, feature0, aux_k_b], dim=1)                
                    flow_b = self.feature_motion_generator(concat_fmap_b)
                    warped_img_f, occ_pred = self.warpnet(flow, flow_b, feature0)
                    
                    warped_image = self.convb(warped_img_f)  # convert to img [N, 3, H, W]
                    warped_image = self.upsample_flow(warped_image, warped_image, bilinear=True)  # to calculate perceptual loss
                    occ_pred = self.upsample_flow(occ_pred, occ_pred, bilinear=True)                    
                    results_dict['occ_pred'] = occ_pred  # to calculate occ loss, [N, 1, H, W]
                   
                    
            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})

        if self.config.DATA.stage == 'ss':
            # calculate the SSL loss
            results_dict['ssl_student'] = student_flow_list
            results_dict['ssl_teacher'] = teacher_flow_list

        if self.loss_config.VGG_LOSS.use:
            if not self.warpnet_config.use:
                warped_image = flow_warp(img0, flow_up)

            x_vgg = self.vgg(img1)
            y_vgg = self.vgg(warped_image)
            results_dict['vgg_src'] = x_vgg
            results_dict['vgg_warped'] = y_vgg
            
        
        return results_dict