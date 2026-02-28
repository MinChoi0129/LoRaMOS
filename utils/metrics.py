import numpy as np


class iouEval:
    def __init__(self, n_classes, ignore=None):
        self.n_classes = n_classes
        self.ignore = ignore if ignore is not None else []
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def addBatch(self, pred, gt):
        valid = (gt >= 0) & (gt < self.n_classes)
        self.conf_matrix += np.bincount(
            self.n_classes * gt[valid] + pred[valid],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)

    def getIoU(self):
        tp = np.diag(self.conf_matrix)
        fp = self.conf_matrix.sum(axis=0) - tp
        fn = self.conf_matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.zeros(self.n_classes)
        valid = denom > 0
        iou[valid] = tp[valid] / denom[valid]
        return iou
