import torch
import torch.nn.functional as F

def focal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    :param pred: (bsz,c,h,w)
    :param target: (bsz,c,h,w)
    :return: (1)
    '''
    pos_mask = target.eq(1).float()
    ne_mask = target.lt(1).float()
    num_pos = pos_mask.sum().int()

    ne_weight = torch.pow(1 - target, 4)

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6) #防止计算log时溢出

    ne_loss = torch.log(1 - pred) * torch.pow(pred, 2) * ne_weight * ne_mask
    ne_loss = ne_loss.sum()

    if num_pos == 0:
        loss = -ne_loss
    else:
        pos_loss = torch.log(pred) * torch.pow((1 - pred), 2) * pos_mask
        pos_loss = pos_loss.sum()
        loss = -(pos_loss + ne_loss) / num_pos

    return loss

def reg_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''
    :param pred: (bsz,2,h,w)
    :param target: (bsz,2,h,w)
    :param mask: (bsz,h,w)
    :return: tensor(size1)
    '''
    mask = torch.unsqueeze(mask, 1).repeat((1,2,1,1))
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss



