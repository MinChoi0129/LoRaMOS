import numpy as np
import torch
from tqdm import tqdm


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


def validate(model, val_loader, range_bins):
    model.eval()
    total_loss, total_mov, total_mbl, n = 0.0, 0.0, 0.0, 0
    moving_evaluator = iouEval(n_classes=3, ignore=[0])
    movable_evaluator = iouEval(n_classes=3, ignore=[0])
    range_evaluators = {r: iouEval(n_classes=3, ignore=[0]) for r in range_bins}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Val", dynamic_ncols=True):
            xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid, _, _ = batch
            xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid = (
                xyzi.cuda(),
                bev_coord.cuda(),
                rv_coord.cuda(),
                rv_input.cuda(),
                label_3d.cuda(),
                label_2d.cuda(),
                num_valid.cuda(),
            )

            out = model(xyzi, bev_coord, rv_coord, rv_input, label_3d, label_2d, num_valid)
            total_loss += out["loss"].item()
            total_mov += out["loss_moving"].item()
            total_mbl += out["loss_movable"].item()
            n += 1

            # Moving evaluation (3D point-wise)
            moving_pred = out["moving_logit_3d"].squeeze(-1).argmax(dim=1).cpu().numpy()
            moving_gt = label_3d.cpu().numpy()
            depth = xyzi[:, -1, 4, :, 0].cpu().numpy()  # [B, N]

            for b in range(moving_pred.shape[0]):
                nv = num_valid[b].item()
                pred_v = moving_pred[b][:nv]
                gt_v = moving_gt[b][:nv].astype(np.int32)
                moving_evaluator.addBatch(pred_v, gt_v)

                depth_v = depth[b][:nv]
                for (rmin, rmax), reval in range_evaluators.items():
                    rmask = (depth_v >= rmin) & (depth_v < rmax)
                    if rmask.any():
                        reval.addBatch(pred_v[rmask], gt_v[rmask])

            # Movable evaluation (2D RV pixel-wise)
            movable_pred = out["movable_logit_2d"].argmax(dim=1).cpu().numpy()
            movable_gt = label_2d.cpu().numpy()
            for b in range(movable_pred.shape[0]):
                movable_evaluator.addBatch(movable_pred[b].flatten(), movable_gt[b].flatten().astype(np.int32))

    _, moving_iou = moving_evaluator.getIoU()
    _, movable_iou = movable_evaluator.getIoU()
    result = {
        "loss": total_loss / n,
        "loss_moving": total_mov / n,
        "loss_movable": total_mbl / n,
        "iou_static": moving_iou[1].item(),
        "iou_moving": moving_iou[2].item(),
        "iou_immovable": movable_iou[1].item(),
        "iou_movable": movable_iou[2].item(),
    }
    for (rmin, rmax), reval in range_evaluators.items():
        _, r_iou = reval.getIoU()
        result[f"iou_moving_{rmin}_{rmax}m"] = r_iou[2].item()
    return result
