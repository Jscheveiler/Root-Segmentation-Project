import torch

def dice_coefficient(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes the Dice coefficient (F1 score) between predictions and targets.
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersect = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersect + eps) / (union + eps)
    return dice.item()

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes the Intersection over Union (IoU) score.
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()

def pixel_accuracy(pred, target, threshold=0.5):
    """
    Computes pixel-wise accuracy between prediction and ground truth.
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

def recall(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes Recall (TP / TP + FN)
    Represents the proportion of actual "Root" pixels that were predicted as "Root".
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    TP = (pred * target).sum()
    FN = ((1 - pred) * target).sum()
    return ((TP + eps) / (TP + FN + eps)).item()

def precision(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes Precision (TP / TP + FP)
    Represents the proportion of predicted "Root" pixels that were actually "Root".
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    return ((TP + eps) / (TP + FP + eps)).item()

def f1_score(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes F1 Score
    It is defined as the harmonic mean of Precision and Recall.
    """
    p = precision(pred, target, threshold, eps)
    r = recall(pred, target, threshold, eps)
    return (2 * p * r + eps) / (p + r + eps)

def specificity(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes Specificity (TN / TN + FP)
    Represents the proportion of actual "Background" pixels that were predicted as "Background".
    (Might not be very useful in the context of the project because of the heavy class imbalance that the dataset presents.)
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    TN = ((1 - pred) * (1 - target)).sum()
    FP = (pred * (1 - target)).sum()
    return ((TN + eps) / (TN + FP + eps)).item()

def balanced_accuracy(pred, target, threshold=0.5, eps=1e-6):
    """
    Computes Balanced Accuracy ((Recall + Specificity) / 2)
    It corresponds to an Accuracy metric that takes class imbalance into consideration
    """
    r = recall(pred, target, threshold, eps)
    s = specificity(pred, target, threshold, eps)
    return (r + s) / 2
