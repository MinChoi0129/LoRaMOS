import numpy as np
import torch


class iouEval:
    def __init__(self, n_classes, ignore=None):
        self.n_classes = n_classes
        self.ignore = torch.tensor(ignore if ignore is not None else []).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]
        ).long()
        self.reset()

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), dtype=torch.long
        )

    def addBatch(self, pred, gt):
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(np.asarray(pred)).long()
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(np.asarray(gt)).long()

        pred_row = pred.reshape(-1)
        gt_row = gt.reshape(-1)

        idxs = torch.stack([pred_row, gt_row], dim=0)
        ones = torch.ones(idxs.shape[-1], dtype=torch.long)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), ones, accumulate=True
        )

    def getStats(self):
        conf = self.conf_matrix.clone().double()
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou
