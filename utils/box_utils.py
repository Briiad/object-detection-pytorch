import torch

def iou(boxes1, boxes2):
    """Compute IoU for two sets of bounding boxes."""
    # Calculate intersection
    inter = (
        torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) -
        torch.max(boxes1[:, None, :2], boxes2[:, :2])
    ).clamp(0).prod(2)

    # Calculate areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2 - inter

    return inter / union

def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        box = boxes[idxs[0]]
        keep.append(idxs[0])
        if idxs.numel() == 1:
            break
        ious = iou(box.unsqueeze(0), boxes[idxs[1:]])
        idxs = idxs[1:][ious.squeeze(0) < iou_threshold]
    return torch.tensor(keep)
