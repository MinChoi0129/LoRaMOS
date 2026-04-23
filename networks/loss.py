import torch
from torch.autograd import Variable
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    # Gradient of the Lovasz extension w.r.t. sorted errors (paper Alg. 1)
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # Cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    # Multi-class Lovasz-Softmax; probas [B, C, H, W], labels [B, H, W] in [0, C-1]
    # classes: 'all' | 'present' | list of class indices
    if ignore is not None:
        valid_mask = labels != ignore
        if valid_mask.sum() == 0:
            return 0

    probas_softmax = F.softmax(probas, dim=1)
    if per_image:
        loss = mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas_softmax, labels)
        )
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas_softmax, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes="present"):
    # Flat Lovasz-Softmax; probas [P, C], labels [P] in [0, C-1]
    if probas.numel() == 0:
        # Only void pixels; gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # Foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    # Flatten batch predictions
    if probas.dim() == 3:
        # Sigmoid-layer output
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 4:
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [P, C] where P = B*H*W
        labels = labels.view(-1)
    else:
        assert probas.dim() == 2

    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


class LovaszSoftmax(torch.nn.Module):
    # nn.Module wrapper for lovasz_softmax with fixed ignore label
    def __init__(self, ignore=0):
        super().__init__()
        self.ignore = ignore

    def forward(self, logits, labels):
        return lovasz_softmax(logits, labels, ignore=self.ignore)


def mean(l, empty=0):
    # Mean compatible with generators
    l = iter(l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
