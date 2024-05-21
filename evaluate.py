import os
import time
import numpy as np
import torch

from data.datasets import FlyingChairs, FlyingThings3D, MpiSintel, KITTI
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile

from utils.utils import InputPadder, compute_out_of_boundary_mask
from loguru import logger

@torch.no_grad()
def create_sintel_submission(model,
                             output_path='sintel_submission',
                             padding_factor=8,
                             save_vis_flow=False,
                             no_save_flo=False,
                             attn_splits_list=None,
                             prop_radius_list=None,
                             ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    output_path = os.path.join(output_path, 'sintel_submission')
    for dstype in ['clean', 'final']:
        test_dataset = MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            results_dict = model(**dict(img0=image1, img1=image2,
                                 attn_splits_list=attn_splits_list,
                                 prop_radius_list=prop_radius_list)
                                 )

            flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, f'frame{(frame+1):04d}.flo')

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not no_save_flo:
                frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

            # Save vis flow
            if save_vis_flow:
                vis_flow_file = output_file.replace('.flo', '.png')
                save_vis_flow_tofile(flow, vis_flow_file)


@torch.no_grad()
def create_kitti_submission(model,
                            output_path='kitti_submission',
                            padding_factor=8,
                            save_vis_flow=False,
                            attn_splits_list=None,
                            prop_radius_list=None,
                            ):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = KITTI(split='testing', aug_params=None)
    output_path = os.path.join(output_path, 'kitti_submission')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        results_dict = model(**dict(img0=image1, img1=image2,
                             attn_splits_list=attn_splits_list,
                             prop_radius_list=prop_radius_list)
                             )

        flow_pr = results_dict['flow_preds'][-1]

        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)

        if save_vis_flow:
            vis_flow_file = output_filename
            save_vis_flow_tofile(flow, vis_flow_file)
        else:
            frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model,
                    with_speed_metric=False,
                    attn_splits_list=False,
                    prop_radius_list=False,
                    ):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    results = {}

    if with_speed_metric:
        s0_10_list = []
        s10_40_list = []
        s40plus_list = []

    val_dataset = FlyingChairs(split='validation')

    logger.info(f'Number of validation image pairs: {len(val_dataset)}')

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # to fix TypeError in replica * on device * when using DataParallel
        results_dict = model(**dict(img0=image1, 
                                    img1=image2, 
                                    attn_splits_list=attn_splits_list,
                                    prop_radius_list=prop_radius_list,))

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        assert flow_pr.size()[-2:] == flow_gt.size()[-2:]

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if with_speed_metric:
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_mask = (flow_gt_speed < 10)
            if valid_mask.max() > 0:
                s0_10_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
            if valid_mask.max() > 0:
                s10_40_list.append(epe[valid_mask].cpu().numpy())

            valid_mask = (flow_gt_speed > 40)
            if valid_mask.max() > 0:
                s40plus_list.append(epe[valid_mask].cpu().numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all > 1)
    px3 = np.mean(epe_all > 3)
    px5 = np.mean(epe_all > 5)
    logger.info(f"Validation Chairs EPE: {epe:.3f}, 1px: {px1:.3f}, 3px: {px3:.3f}, 5px: {px5:.3f}")
    results['chairs_epe'] = epe
    results['chairs_1px'] = px1
    results['chairs_3px'] = px3
    results['chairs_5px'] = px5

    if with_speed_metric:
        s0_10 = np.mean(np.concatenate(s0_10_list))
        s10_40 = np.mean(np.concatenate(s10_40_list))
        s40plus = np.mean(np.concatenate(s40plus_list))

        logger.info(f"Validation Chairs s0_10: {s0_10:.3f}, s10_40: {s10_40:.3f}, s40+: {s40plus:.3f}")

        results['chairs_s0_10'] = s0_10
        results['chairs_s10_40'] = s10_40
        results['chairs_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_things(model,
                    padding_factor=8,
                    with_speed_metric=True,
                    max_val_flow=400,
                    val_things_clean_only=True,
                    attn_splits_list=False,
                    prop_radius_list=False,
                    ):
    """ Peform validation using the Things (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        if val_things_clean_only:
            if dstype == 'frames_finalpass':
                continue

        val_dataset = FlyingThings3D(dstype=dstype, test_set=True, validate_subset=True,
                                          )
        logger.info(f'Number of validation image pairs: {len(val_dataset)}')
        epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            results_dict = model(**dict(img0=image1, img1=image2,
                                 attn_splits_list=attn_splits_list,
                                 prop_radius_list=prop_radius_list)
                                 )
            flow_pr = results_dict['flow_preds'][-1]

            flow = padder.unpad(flow_pr[0]).cpu()

            # Evaluation on flow <= max_val_flow
            flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            valid_gt = valid_gt * (flow_gt_speed < max_val_flow)
            valid_gt = valid_gt.contiguous()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            val = valid_gt >= 0.5
            epe_list.append(epe[val].cpu().numpy())

            if with_speed_metric:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_list = np.concatenate(epe_list)

        epe = np.mean(epe_list)
        px1 = np.mean(epe_list>1)
        px3 = np.mean(epe_list>3)
        px5 = np.mean(epe_list>5)

        if dstype == 'frames_cleanpass':
            dstype = 'things_clean'
        if dstype == 'frames_finalpass':
            dstype = 'things_final'

        logger.info(f"Validation Things test set ({dstype}) EPE: {epe:.3f}, 1px: {px1:.3f}, 3px: {px3:.3f}, 5px: {px5:.3f}")
        results[dstype + '_epe'] = epe
        results[dstype + '_1px'] = px1
        results[dstype + '_3px'] = px3
        results[dstype + '_5px'] = px5

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            logger.info(f"Validation Things test ({dstype}) s0_10: {s0_10:.3f}, s10_40: {s10_40:.3f}, s40+: {s40plus:.3f}")

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

    return results


@torch.no_grad()
def validate_sintel(model,
                    count_time=False,
                    padding_factor=8,
                    with_speed_metric=False,
                    evaluate_matched_unmatched=False,
                    attn_splits_list=False,
                    prop_radius_list=False,
                    ):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    if count_time:
        total_time = 0
        num_runs = 100

    for dstype in ['clean', 'final']:
        val_dataset = MpiSintel(split='training', dstype=dstype,
                                     load_occlusion=evaluate_matched_unmatched,
                                     )

        logger.info(f'Number of validation image pairs: {len(val_dataset)}')
        epe_list = []

        if evaluate_matched_unmatched:
            matched_epe_list = []
            unmatched_epe_list = []

        if with_speed_metric:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []

        for val_id in range(len(val_dataset)):
            if evaluate_matched_unmatched:
                image1, image2, flow_gt, valid, noc_valid = val_dataset[val_id]

                # compuate in-image-plane valid mask
                in_image_valid = compute_out_of_boundary_mask(flow_gt.unsqueeze(0)).squeeze(0)  # [H, W]

            else:
                image1, image2, flow_gt, _ = val_dataset[val_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, padding_factor=padding_factor)
            image1, image2 = padder.pad(image1, image2)

            if count_time and val_id >= 5:  # 5 warmup
                torch.cuda.synchronize()
                time_start = time.perf_counter()

            results_dict = model(**dict(img0=image1, img1=image2,
                                 attn_splits_list=attn_splits_list,
                                 prop_radius_list=prop_radius_list)
                                 )

            # useful when using parallel branches
            flow_pr = results_dict['flow_preds'][-1]

            if count_time and val_id >= 5:
                torch.cuda.synchronize()
                total_time += time.perf_counter() - time_start

                if val_id >= num_runs + 4:
                    break

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            if evaluate_matched_unmatched:
                matched_valid_mask = (noc_valid > 0.5) & (in_image_valid > 0.5)

                if matched_valid_mask.max() > 0:
                    matched_epe_list.append(epe[matched_valid_mask].cpu().numpy())
                    unmatched_epe_list.append(epe[~matched_valid_mask].cpu().numpy())

            if with_speed_metric:
                flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
                valid_mask = (flow_gt_speed < 10)
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all > 1)
        px3 = np.mean(epe_all > 3)
        px5 = np.mean(epe_all > 5)

        dstype_ori = dstype

        logger.info(f"Validation Sintel ({dstype}) EPE: {epe:.3f}, 1px: {px1:.3f}, 3px: {px3:.3f}, 5px: {px5:.3f}")

        dstype = 'sintel_' + dstype

        results[dstype + '_epe'] = np.mean(epe_list)
        results[dstype + '_1px'] = px1
        results[dstype + '_3px'] = px3
        results[dstype + '_5px'] = px5

        if with_speed_metric:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))

            logger.info(f"Validation Sintel ({dstype_ori}) s0_10: {s0_10:.3f}, s10_40: {s10_40:.3f}, s40+: {s40plus:.3f}")

            results[dstype + '_s0_10'] = s0_10
            results[dstype + '_s10_40'] = s10_40
            results[dstype + '_s40+'] = s40plus

        if count_time:
            logger.info(f'Time: {(total_time/num_runs):.6f}s')
            break  # only the clean pass when counting time

        if evaluate_matched_unmatched:
            matched_epe = np.mean(np.concatenate(matched_epe_list))
            unmatched_epe = np.mean(np.concatenate(unmatched_epe_list))

            logger.info(f'Validatation Sintel ({dstype_ori}) matched epe: {matched_epe:.3f}, unmatched epe: {unmatched_epe:.3f}')

            results[dstype + '_matched'] = matched_epe
            results[dstype + '_unmatched'] = unmatched_epe

    return results


@torch.no_grad()
def validate_kitti(model,
                   padding_factor=8,
                   with_speed_metric=True,
                   average_over_pixels=True,
                   attn_splits_list=False,
                   prop_radius_list=False,
                   ):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()

    val_dataset = KITTI(split='training')
    logger.info(f'Number of validation image pairs: {len(val_dataset)}')

    out_list, epe_list = [], []
    out_epe_list = []
    results = {}

    if with_speed_metric:
        if average_over_pixels:
            s0_10_list = []
            s10_40_list = []
            s40plus_list = []
        else:
            s0_10_epe_sum = 0
            s0_10_valid_samples = 0
            s10_40_epe_sum = 0
            s10_40_valid_samples = 0
            s40plus_epe_sum = 0
            s40plus_valid_samples = 0

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti', padding_factor=padding_factor)
        image1, image2 = padder.pad(image1, image2)

        results_dict = model(**dict(img0=image1, img1=image2,
                             attn_splits_list=attn_splits_list,
                             prop_radius_list=prop_radius_list)
                             )

        # useful when using parallel branches
        flow_pr = results_dict['flow_preds'][-1]

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        if with_speed_metric:
            # flow_gt_speed = torch.sum(flow_gt ** 2, dim=0).sqrt()
            flow_gt_speed = mag

            if average_over_pixels:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_list.append(epe[valid_mask].cpu().numpy())

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_list.append(epe[valid_mask].cpu().numpy())

            else:
                valid_mask = (flow_gt_speed < 10) * (valid_gt >= 0.5)  # note KITTI GT is sparse
                if valid_mask.max() > 0:
                    s0_10_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s0_10_valid_samples += 1

                valid_mask = (flow_gt_speed >= 10) * (flow_gt_speed <= 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s10_40_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s10_40_valid_samples += 1

                valid_mask = (flow_gt_speed > 40) * (valid_gt >= 0.5)
                if valid_mask.max() > 0:
                    s40plus_epe_sum += (epe * valid_mask).sum() / valid_mask.sum()
                    s40plus_valid_samples += 1

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        out_epe = (epe > 3.0).float()

        if average_over_pixels:
            epe_list.append(epe[val].cpu().numpy())
        else:
            epe_list.append(epe[val].mean().item())

        out_list.append(out[val].cpu().numpy())
        out_epe_list.append(out_epe[val].cpu().numpy())

    if average_over_pixels:
        epe_list = np.concatenate(epe_list)
    else:
        epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    out_epe_list = np.concatenate(out_epe_list)

    epe = np.mean(epe_list)
    px1 = np.mean(epe_list > 1)
    px3 = np.mean(epe_list > 3)
    px5 = np.mean(epe_list > 5)
    
    f1 = 100 * np.mean(out_list)
    f1_epe = 100 * np.mean(out_epe_list)
    
    logger.info(f"Validation KITTI EPE: {epe:.3f}, 1px: {px1:.3f}, 3px: {px3:.3f}, 5px: {px5:.3f}, F1_all: {f1:.3f}, F1_epe: {f1_epe:.3f}")

    if with_speed_metric:
        if average_over_pixels:
            s0_10 = np.mean(np.concatenate(s0_10_list))
            s10_40 = np.mean(np.concatenate(s10_40_list))
            s40plus = np.mean(np.concatenate(s40plus_list))
        else:
            s0_10 = s0_10_epe_sum / s0_10_valid_samples
            s10_40 = s10_40_epe_sum / s10_40_valid_samples
            s40plus = s40plus_epe_sum / s40plus_valid_samples

        logger.info(f"Validation KITTI s0_10: {s0_10:.3f}, s10_40: {s10_40:.3f}, s40+: {s40plus:.3f}")

        results['kitti_s0_10'] = s0_10
        results['kitti_s10_40'] = s10_40
        results['kitti_s40+'] = s40plus
    results['kitti_epe'] = epe
    results['kitti_1px'] = px1
    results['kitti_3px'] = px3
    results['kitti_5px'] = px5
    results['kitti_f1all'] = f1
    results['kitti_f1epe'] = f1_epe
    return results

@torch.no_grad()
def evaluate(model, config):
    val_results = {}

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
        
@torch.no_grad()
def submission(model, config):
    logger.info(f'Creating submission for {config.VALIDATE.val_dataset}')
    # NOTE: val_dataset is a list
    for val_ds in config.VALIDATE.val_dataset:
        if val_ds == 'sintel':
            logger.info(f'Creating submission for Sintel')
            create_sintel_submission(model,
                                    output_path=config.SUBMISSION.output_path,
                                    padding_factor=config.MODEL.padding_factor,
                                    no_save_flo=config.SUBMISSION.no_save_flo,
                                    attn_splits_list=config.MODEL.attn_splits_list,
                                    prop_radius_list=config.MODEL.prop_radius_list,
                                    )
        elif val_ds == 'kitti':
            logger.info(f'Creating submission for KITTI')
            create_kitti_submission(model,
                                    output_path=config.SUBMISSION.output_path,
                                    padding_factor=config.MODEL.padding_factor,
                                    attn_splits_list=config.MODEL.attn_splits_list,
                                    prop_radius_list=config.MODEL.prop_radius_list,
                                    )
        else:
            logger.error(f'Not supported dataset for submission: {val_ds}')
            continue
    logger.info(f'Submission Done.')
