import torch
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

from data.datasets import build_train_dataset
from utils.loss import (flow_loss_func, self_supervised_loss, calculate_perceptual_loss, calculate_occlusion_loss)
from utils.utils import (save_dst_ckpt, get_loss_weight)
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti)

from loguru import logger



def train(model, config, optimizer, start_epoch=0, start_step=0, device='cuda'):
    # training datset
    train_dataset = build_train_dataset(config)
    if config.SYSTEM.local_rank == 0:
        logger.info(f'Number of training images: {len(train_dataset)}')

    shuffle = True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN.batch_size,
                                               shuffle=shuffle, num_workers=config.TRAIN.num_workers,
                                               pin_memory=True, drop_last=True)

    last_epoch = start_step if config.RESUME.use and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config.TRAIN.lr,
        config.TRAIN.num_steps + 10,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos',
        last_epoch=last_epoch,
    )

    if config.SYSTEM.local_rank == 0:
        tb_dir = osp.join(config.SYSTEM.checkpoint_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        summary_writer = SummaryWriter(tb_dir)
        
    total_steps = start_step
    epoch = start_epoch
    logger.info('Start training')
    training_epe = 0.0
    delta_steps = 0
    loss_config = config.TRAIN.LOSS
    while total_steps < config.TRAIN.num_steps:
        model.train()

        for i, sample in enumerate(train_loader):
            if config.DATA.stage == 'ss':  # self-supervised
                img0, img1 = [x.to(device) for x in sample]
            elif config.DATA.stage == 'sintel_occ':  # use sintel occlusion maps
                img0, img1, flow_gt, valid, noc_valid = [x.to(device) for x in sample]  # `noc_valid` represents the occluded mask, 0 is occluded, 1 is non-occluded
            else:
                img0, img1, flow_gt, valid = [x.to(device) for x in sample]


            results_dict = model(img0=img0, img1=img1,
                                 attn_splits_list=config.MODEL.attn_splits_list,
                                 prop_radius_list=config.MODEL.prop_radius_list)

            flow_preds = results_dict['flow_preds']
            loss_weight = get_loss_weight(total_steps, config.TRAIN.num_steps, lr_scheduler.get_last_lr()[0])
            
            if config.DATA.stage == 'ss':
                student_flow_list = results_dict['ssl_student']
                teacher_flow_list = results_dict['ssl_teacher']
                loss, metrics = self_supervised_loss(student_flow_list, teacher_flow_list)
                if loss_config.VGG_LOSS.use:
                    vgg_warped = results_dict['vgg_warped']
                    vgg_src = results_dict['vgg_src']
                    perceptual_loss = calculate_perceptual_loss(vgg_src, vgg_warped, loss_weight)
                    loss += perceptual_loss                
            else:
                loss, metrics = flow_loss_func(flow_preds, flow_gt, valid,
                                               gamma=loss_config.L1_LOSS.gamma,
                                               max_flow=loss_config.L1_LOSS.max_flow,
                                               )
                if loss_config.VGG_LOSS.use:
                    vgg_warped = results_dict['vgg_warped']
                    vgg_src = results_dict['vgg_src']
                    perceptual_loss = calculate_perceptual_loss(vgg_src, vgg_warped, loss_weight)
                    loss += perceptual_loss
                
                if 'occ_pred' in results_dict.keys():  # sintel
                    # occ loss
                    occ_pred = results_dict['occ_pred']  # [H, W]
                    occ_loss = calculate_occlusion_loss(noc_valid, occ_pred, loss_weight)
                    loss += occ_loss
                

            if isinstance(loss, float):
                continue

            if torch.isnan(loss):
                continue

            training_epe += metrics['epe']

            # more efficient zero_grad
            for param in model.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.grad_clip)

            optimizer.step()

            lr_scheduler.step()
            total_steps += 1
            delta_steps += 1

            if total_steps % config.TRAIN.print_freq == 0 and config.SYSTEM.local_rank == 0:
                logger.info(f'train [{total_steps}/{config.TRAIN.num_steps} step] lr={lr_scheduler.get_last_lr()[0]:.6f}, loss={loss:.3f}, epe={metrics["epe"]:.3f}, 1px(epe>1)={metrics["1px"]:.3f}, 3px(epe>3)={metrics["3px"]:.3f}, 5px(epe>5)={metrics["5px"]:.3f}')
                summary_writer.add_scalar('loss', loss, total_steps)
                summary_writer.add_scalar('epe', metrics["epe"], total_steps)
                summary_writer.add_scalar('1px', metrics["1px"], total_steps)
                summary_writer.add_scalar('3px', metrics["3px"], total_steps)
                summary_writer.add_scalar('5px', metrics["5px"], total_steps)
                summary_writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], total_steps)


            if total_steps % config.TRAIN.save_ckpt_freq == 0 or total_steps == config.TRAIN.num_steps:
                if config.SYSTEM.local_rank == 0:
                    checkpoint_path = os.path.join(config.SYSTEM.checkpoint_dir, f'step_{total_steps:06d}.pth')
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if total_steps % config.TRAIN.save_latest_ckpt_freq == 0:
                checkpoint_path = os.path.join(config.SYSTEM.checkpoint_dir, 'checkpoint_latest.pth')

                if config.SYSTEM.local_rank == 0:
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)

            if (total_steps % config.VALIDATE.val_freq == 0 or total_steps == config.TRAIN.num_steps) and config.SYSTEM.local_rank == 0:
                logger.info('Start validation')

                val_results = {}
                # support validation on multiple datasets
                if 'chairs' in config.VALIDATE.val_dataset:
                    results_dict = validate_chairs(model,
                                                   with_speed_metric=config.VALIDATE.with_speed_metric,
                                                   attn_splits_list=config.MODEL.attn_splits_list,
                                                   prop_radius_list=config.MODEL.prop_radius_list,
                                                   )
                    val_results.update(results_dict)

                if 'things' in config.VALIDATE.val_dataset:
                    results_dict = validate_things(model,
                                                   padding_factor=config.MODEL.padding_factor,
                                                   with_speed_metric=config.VALIDATE.with_speed_metric,
                                                   attn_splits_list=config.MODEL.attn_splits_list,
                                                   prop_radius_list=config.MODEL.prop_radius_list,
                                                   )
                    val_results.update(results_dict)
                    

                if 'sintel' in config.VALIDATE.val_dataset:
                    results_dict = validate_sintel(model,
                                                   padding_factor=config.MODEL.padding_factor,
                                                   with_speed_metric=config.VALIDATE.with_speed_metric,
                                                   evaluate_matched_unmatched=config.VALIDATE.evaluate_matched_unmatched,
                                                   attn_splits_list=config.MODEL.attn_splits_list,
                                                   prop_radius_list=config.MODEL.prop_radius_list,
                                                   )
                    val_results.update(results_dict)
                    

                if 'kitti' in config.VALIDATE.val_dataset:
                    results_dict = validate_kitti(model,
                                                  padding_factor=config.MODEL.padding_factor,
                                                  with_speed_metric=config.VALIDATE.with_speed_metric,
                                                  attn_splits_list=config.MODEL.attn_splits_list,
                                                  prop_radius_list=config.MODEL.prop_radius_list,
                                                  )
                    val_results.update(results_dict)

                model.train()

            if total_steps >= config.TRAIN.num_steps:
                save_dst_ckpt(model, config.DATA.stage, config.SYSTEM.final_ckpt_dir)
                logger.info('Training done')
                return

        epoch += 1

