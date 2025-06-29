import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-6):
    """
    Computes Dice Loss. Assumes pred are logits.
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()

def bce_dice_loss(pred, target, bce_weight=0.7):
    """
    Hybrid loss combining BCE and Dice Loss.
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)

    return bce_weight * bce + (1 - bce_weight) * dice
