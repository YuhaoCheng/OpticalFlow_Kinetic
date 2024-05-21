import torch


def flow_loss_func(flow_preds, flow_ref, valid=None,
                   gamma=0.9,
                   max_flow=400,
                   **kwargs,
                   ):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_ref ** 2, dim=1).sqrt()  # [B, H, W]
    if valid is not None:
        valid = (valid >= 0.5) & (mag < max_flow)
    else:
        valid = (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        i_loss = (flow_preds[i] - flow_ref).abs()

        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_ref) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics



def self_supervised_loss(flow_preds, flow_refs, gamma=0.8, loss_func='L1', loss_weight=1.0, max_flow=400):
    
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        mag = torch.sum(flow_refs[i]**2, dim=1).sqrt()
        valid = (mag < max_flow)
    
        i_weight = gamma**(n_predictions - i - 1)
        if loss_func == 'L2': 
            i_loss = torch.nn.MSELoss()(flow_preds[i], flow_refs[i]*0.5) * loss_weight
        elif loss_func == 'S_L1':
            i_loss = torch.nn.SmoothL1Loss()(flow_preds[i], flow_refs[i]*0.5) * loss_weight
        else:  # L1 loss
            i_loss = abs(flow_preds[i] - flow_refs[i]*0.5) * loss_weight
        flow_loss += (i_weight * (valid[:, None]*i_loss)).mean()
    flow_loss = flow_loss.mean()
    epe = torch.sum((flow_preds[-1] - flow_refs[-1]*0.5) ** 2, dim=1).sqrt()

    if valid.max() < 0.5:
        pass

    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss, metrics


def calculate_perceptual_loss(src, warped, loss_weight):
    perceptual_loss = loss_weight * torch.abs(warped[i] - src[i].detach()).mean()
    return perceptual_loss


def calculate_occlusion_loss(occ_gt, occ_pred, loss_weight):
    occ_loss = loss_weight *  torch.abs(occ_gt - occ_pred).mean()
    return occ_loss
    