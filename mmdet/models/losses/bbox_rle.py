import torch
import torch.nn as nn
from torch import distributions
from ..builder import LOSSES
import math

from mmdet.ops import box_iou_rotated_differentiable


@LOSSES.register_module()
class BoxRLELoss(nn.Module):
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
                 q_distribution='laplace',
                 version='v1'):
        super(BoxRLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_distribution = q_distribution
        self.version = version
        self.flow_model = CateSpecificRealNVP()

    def forward(self, pred, target, sigma, labels, target_weight=None):
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
        index = target_weight.mean(1).bool()
        if index.sum() == 0:
            return torch.tensor([0]).to(pred)
        labels = labels[index]
        sigma = (1 - sigma.sigmoid())[index]
        pred = pred[index]
        target = target[index]
        bbox_distance = box_iou_rotated_differentiable(pred, target, version=self.version).clamp(min=1e-6)
        bbox_distance = 1-bbox_distance
        error = (bbox_distance) / (sigma + 1e-9)
        # (B, K, 2)
        direct = (pred[:, :2] - target[:, :2])/(pred[:, :2] - target[:, :2]).norm(dim=1, keepdim=True)
        dir_error = error[..., None] * direct
        log_phi = self.flow_model.log_prob(dir_error.reshape(-1, 2), labels)
        log_phi = log_phi.reshape(target.shape[0], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], 1)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_distribution in ['laplace', 'gaussian']
            if self.q_distribution == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error ** 2

            loss = nf_loss.squeeze(-1) + loss_q
        else:
            loss = nf_loss.squeeze(-1)
        # if self.use_target_weight:
        #     assert target_weight is not None
        #     loss *= target_weight
        if self.size_average:
            loss /= len(loss)

        return loss.sum() * 0.1


class CateSpecificRealNVP(nn.Module):
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
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def get_trans_net():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(), nn.Linear(64, 64),
            nn.LeakyReLU(), nn.Linear(64, 2))

    def prior(self, label):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc[label], self.cov[label])

    def __init__(self, num_class=15):
        super(CateSpecificRealNVP, self).__init__()

        self.register_buffer('loc', nn.Parameter(torch.zeros((num_class, 2))))
        self.register_buffer('cov', nn.Parameter(torch.eye(2)[None].repeat(num_class, 1, 1)))
        # self.loc = nn.Parameter(torch.zeros((num_class, 2)))
        # self.cov = nn.Parameter(torch.eye(2)[None].repeat(num_class, 1, 1))
        self.register_buffer(
            'mask', torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList(
            [self.get_scale_net() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList(
            [self.get_trans_net() for _ in range(len(self.mask))])
        self.num_class = num_class
        # self.Normal_list = []
        # for i in range(self.num_class):
        #     self.Normal_list.append(distributions.MultivariateNormal(self.loc[i], self.cov[i]))
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

    def log_prob(self, x, label):
        """Calculate the log probability of given sample in data space."""
        z, log_det = self.backward_p(x)
        return self.prior(label).log_prob(z) + log_det
