import torch
import torch.nn as nn
from torch import distributions
from ..builder import LOSSES
from .utils import weighted_loss
import math

@weighted_loss
def smooth_l1_loss_contiguous(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    pred_bbox = pred[:, :-1]
    target_bbox = target[:, :-1]
    pred_theta = pred[:, [-1]]
    target_theta = target[:, [-1]]
    diff = torch.abs(pred_bbox - target_bbox)
    diff_sin = torch.abs(torch.sin(pred_theta) - torch.sin(target_theta))
    diff_cos = torch.abs(torch.cos(pred_theta) - torch.cos(target_theta))
    diff = torch.cat([diff, diff_sin+diff_cos], dim=1)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@LOSSES.register_module()
class SmoothL1LossContiguous(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1LossContiguous, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss_contiguous(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=True,
                 size_average=True,
                 residual=True,
                 q_distribution='laplace'):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution

        self.flow_model = RealNVP()

    def forward(self, pred,  target, sigma, target_weight=None):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            pred (Tensor[N, K, D]): Output regression.
            sigma (Tensor[N, K, D]): Output sigma.
            target (Tensor[N, K, D]): Target regression.
            target_weight (Tensor[N, K, D]):
                Weights across different joint types.
        """
        # sigma = sigma.sigmoid()
        # sigma = sigma.repeat(1, 2)
        # pred_sin = torch.sin(pred)
        # pred_cos = torch.cos(pred)
        # target_sin = torch.sin(target)
        # target_cos = torch.cos(target)
        #
        # pred_ = torch.cat([pred_sin, pred_cos], dim=1)
        # target_ = torch.cat([target_sin, target_cos], dim=1)
        # error = (pred_ - target_) / (sigma + 1e-9)
        # # (B, K, 2)
        # log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        # log_phi = log_phi.reshape(target_.shape[0], 1)
        # log_sigma = torch.log(sigma).reshape(target_.shape[0], 2)
        # nf_loss = log_sigma - log_phi
        #
        # if self.residual:
        #     assert self.q_distribution in ['laplace', 'gaussian']
        #     if self.q_distribution == 'laplace':
        #         loss_q = torch.log(sigma * 2) + torch.abs(error)
        #     else:
        #         loss_q = torch.log(
        #             sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2
        #
        #     loss = nf_loss + loss_q
        # else:
        #     loss = nf_loss
        #
        # loss = loss.sum(1)
        # if self.use_target_weight:
        #     assert target_weight is not None
        #     loss *= target_weight
        #
        # if self.size_average:
        #     loss /= len(loss)
        #
        # return loss.sum()
        sigma =(1 - sigma.sigmoid())
        error = (pred - target) / (sigma[..., None] + 1e-9)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 5))
        log_phi = log_phi.reshape(target.shape[0], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], 1)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2)[..., None] + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi))[..., None] + 0.5 * error ** 2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss
        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum() * 0.001


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the official implementation of RLE
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    See also `real-nvp-pytorch
    <https://github.com/senya-ashukha/real-nvp-pytorch>`_.
    """

    @staticmethod
    def get_scale_net():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(5, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 5), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(5, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 5))

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super(RealNVP, self).__init__()

        self.register_buffer('loc', torch.zeros(5))
        self.register_buffer('cov', torch.eye(5))
        self.register_buffer(
            'mask', torch.tensor([[0,0,1,1, 1], [1, 1,0 , 0, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList(
            [self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det
