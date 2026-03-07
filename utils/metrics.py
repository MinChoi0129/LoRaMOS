import numpy as np


class iouEval:
    def __init__(self, n_classes, ignore=None):
        self.n_classes = n_classes
        self.ignore = set(ignore) if ignore is not None else set()
        self.include = [n for n in range(n_classes) if n not in self.ignore]
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def addBatch(self, pred, gt):
        pred = np.asarray(pred, dtype=np.int64).ravel()
        gt = np.asarray(gt, dtype=np.int64).ravel()
        np.add.at(self.conf_matrix, (pred, gt), 1)

    def getStats(self):
        conf = self.conf_matrix.astype(np.float64).copy()
        for ig in self.ignore:
            conf[ig, :] = 0
            conf[:, ig] = 0
        tp = np.diag(conf)
        fp = conf.sum(axis=1) - tp
        fn = conf.sum(axis=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        union = tp + fp + fn + 1e-15
        iou = tp / union
        iou_mean = np.mean(iou[self.include])
        return iou_mean, iou
