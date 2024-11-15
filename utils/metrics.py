import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class DetectionMetrics:
    def __init__(self, iou_thresholds=None):
        """
        Initialize the detection metrics.
        Args:
            iou_thresholds (list or None): List of IoU thresholds (e.g., [0.5, 0.75]). 
                                           Default is [0.5:0.95] if None.
        """
        self.map_metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)

    def update(self, preds, targets):
        """
        Update the metrics with predictions and targets.
        Args:
            preds (list[dict]): Predictions as [{boxes, scores, labels}].
            targets (list[dict]): Targets as [{boxes, labels}].
        """
        self.map_metric.update(preds, targets)

    def compute(self):
        """
        Compute the accumulated metrics.
        Returns:
            dict: Computed metrics (e.g., mAP, precision, recall).
        """
        return self.map_metric.compute()

    def reset(self):
        """Reset the metrics for the next epoch."""
        self.map_metric.reset()
