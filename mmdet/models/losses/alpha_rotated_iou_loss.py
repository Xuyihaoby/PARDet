import torch
import torch.nn as nn
from mmdet.ops import box_iou_rotated_differentiable

from mmdet.models.builder import LOSSES
from .utils import weighted_loss
from mmdet.core.bbox.rtransforms import enclosing_box


@weighted_loss
def alpha_iou_loss(pred, target, alpha=3, linear=False, eps=1e-6, version='v1'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = box_iou_rotated_differentiable(pred, target, version=version).clamp(min=eps)
    if linear:
        loss = 1 - ious ** alpha
    else:
        loss = -ious.log()
    return loss


@LOSSES.register_module()
class AlphaRotatedIoULoss(nn.Module):

    def __init__(self, alpha=3, linear=True, eps=1e-6, reduction='mean', loss_weight=1.0, version='v1'):
        super(AlphaRotatedIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version
        self.alpha = alpha

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * alpha_iou_loss(
            pred,
            target,
            weight,
            linear=self.linear,
            version=self.version,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            alpha=self.alpha,
            **kwargs)
        return loss


@weighted_loss
def alpha_giou_loss(pred, target, alpha=3, eps=1e-6, version='v1', enclosing='smallest'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious, corners1, corners2, u = box_iou_rotated_differentiable(pred, target, version=version, iou_only=False)
    ious = ious.clamp(min=eps)
    u.clamp(min=eps)
    w, h = enclosing_box(corners1, corners2, enclosing)
    area_c = w * h
    giou = ious**alpha - ((area_c - u) / area_c)**alpha
    loss = 1 - giou
    return loss


@LOSSES.register_module()
class AlphaRotatedGIoULoss(nn.Module):

    def __init__(self, alpha=3, eps=1e-6, reduction='mean', loss_weight=1.0, version='v1', enclosing='smallest'):
        super(AlphaRotatedGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version
        self.alpha = alpha
        self.enclosing = enclosing

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * alpha_giou_loss(
            pred,
            target,
            weight,
            version=self.version,
            eps=self.eps,
            enclosing=self.enclosing,
            reduction=reduction,
            avg_factor=avg_factor,
            alpha=self.alpha,
            **kwargs)
        return loss

@weighted_loss
def alpha_diou_loss(pred, target, alpha=3, eps=1e-6, version='v1', enclosing='smallest'):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
            shape (n, 5).
        target (Tensor): Corresponding gt bboxes, shape (n, 5).
        linear (bool):  If True, use linear scale of loss instead of
            log scale. Default: False.
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious, corners1, corners2, u = box_iou_rotated_differentiable(pred, target, version=version, iou_only=False)
    ious = ious.clamp(min=eps)
    u.clamp(min=eps)
    w, h = enclosing_box(corners1, corners2, enclosing)
    c2 = w * w + h * h  # (B, N)
    x_offset = pred[..., 0] - target[..., 0]
    y_offset = pred[..., 1] - target[..., 1]
    d2 = x_offset * x_offset + y_offset * y_offset
    diou_loss = 1. - ious**alpha + d2**alpha / c2**alpha
    return diou_loss


@LOSSES.register_module()
class AlphaRotatedDIoULoss(nn.Module):

    def __init__(self, alpha=3, eps=1e-6, reduction='mean', loss_weight=1.0, version='v1', enclosing='smallest'):
        super(AlphaRotatedDIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.version = version
        self.alpha = alpha
        self.enclosing = enclosing

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * alpha_diou_loss(
            pred,
            target,
            weight,
            version=self.version,
            eps=self.eps,
            enclosing=self.enclosing,
            reduction=reduction,
            avg_factor=avg_factor,
            alpha=self.alpha,
            **kwargs)
        return loss
