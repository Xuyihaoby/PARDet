import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.core import (images_to_levels, multi_apply, unmap, ranchor_inside_flags,
                        multiclass_nms_r, reduce_mean)

from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from mmcv.runner import force_fp32
from mmdet.core.bbox import obb2poly, poly2obb
from mmdet.ops import minaerarect, pointsJf, pointsJfAlign

import torch.nn.functional as F
from mmdet.core.bbox.utils import GaussianMixture


@HEADS.register_module()
class PARDetHeadATSS(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 n_components=6,
                 gradient_mul=0,
                 stacked_convs=4,
                 point_num=60,
                 conv_cfg=None,
                 norm_cfg=None,
                 angle_version='v2',
                 anchor_generator=dict(
                     type='RAnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128],
                     angles=None),
                 loss_pt=dict(type='KLDRepPointsLoss',
                              loss_weight=1.0,
                              num_points=60),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_num = point_num
        self.gradient_mul = gradient_mul
        super(PARDetHeadATSS, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)
        self.gmm = GaussianMixture(n_components=n_components)
        self.loss_pt = build_loss(loss_pt)
        self.angle_version = angle_version

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.pt_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.pt_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg
                )
            )
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.retina_pt = nn.Conv2d(
            self.feat_channels, self.num_anchors * self.point_num * 2, 3, padding=1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        for m in self.pt_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

        normal_init(self.retina_pt, std=0.01)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        if not self.training:
            return cls_score, bbox_pred


        pt_feat = x
        for pt_conv in self.pt_convs:
            pt_feat = pt_conv(pt_feat)
        pt_pred = self.retina_pt(pt_feat)
        return cls_score, bbox_pred, pt_pred

    def loss_single(self, cls_score, bbox_pred, pt_pred, anchors, anchor_points, labels, label_weights,
                    bbox_targets, bbox_weights, stride, num_total_samples):
        # each lvls of all batches
        # classification loss
        n, _, h, w = bbox_pred.size()

        per_lvl_index = h * w
        each_img_pos_index = []

        # 得到索引
        bbox_weights = bbox_weights.reshape(-1, 5)  # [batchsize, num_bboxes, 5]
        pos_index = bbox_weights.mean(dim=1).nonzero(as_tuple=False).squeeze(1)
        if len(pos_index) == 0:
            labels = labels.reshape(-1)
            label_weights = label_weights.reshape(-1)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, avg_factor=num_total_samples)
            loss_pt = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            return loss_cls, loss_bbox, loss_pt

        for idx in range(n):
            each_img_pos_index.append((pos_index >= idx * per_lvl_index) & (pos_index < (idx + 1) * per_lvl_index))

        each_img_pos_index = torch.stack(each_img_pos_index)[..., None].repeat(1, 1, self.gmm.n_components).reshape(
            n, -1)

        _pt_pred_semi_detach = pt_pred.detach()
        _temp_offset = anchor_points.reshape(n, h, w, -1).permute(0, 3, 1, 2) / stride[0] + _pt_pred_semi_detach

        _temp_pos_anchor = anchor_points.reshape(-1, self.point_num * 2)[pos_index]
        _temp_scaled_pos_center = torch.stack(
            (_temp_pos_anchor[:, 0::2].mean(dim=-1), _temp_pos_anchor[:, 1::2].mean(dim=-1)),
            dim=-1) / stride[0]
        _temp_scaled_pos_center_set = _temp_scaled_pos_center[:, None, :].repeat(1, self.point_num, 1)


        _minus_pos_set = \
        _temp_offset.reshape(n, self.point_num * 2, -1).transpose(-1, -2).reshape(-1, self.point_num, 2)[
            pos_index] - _temp_scaled_pos_center_set
        self.gmm.fit(_minus_pos_set)

        _label_pos = labels.reshape(-1)[pos_index][..., None].repeat(1, self.gmm.n_components).reshape(-1)
        _score_pos = self.gmm.pi.squeeze(-1).reshape(-1).detach()
        _center_pos = self.gmm.mu.reshape(-1, 2).detach() + _temp_scaled_pos_center[:, None, :].repeat(1, self.gmm.n_components,
                                                                                                1).reshape(-1, 2)
        _center_pos_offset = (2 * _center_pos / (w - 1) - 1).unsqueeze(0)[None].repeat(n, 1, 1, 1)
        t_cls_score = F.grid_sample(cls_score, _center_pos_offset.detach(), padding_mode='border', align_corners=True)
        t_bbox_pred = F.grid_sample(bbox_pred, _center_pos_offset.detach(), padding_mode='border', align_corners=True)

        _sift_value = (t_cls_score.squeeze(2) * each_img_pos_index.unsqueeze(1)).permute(0, 2, 1)
        _sift_value = _sift_value[_sift_value.nonzero(as_tuple=True)].reshape(-1, 15)

        _sift_loc = (t_bbox_pred.squeeze(2) * each_img_pos_index.unsqueeze(1)).permute(0, 2, 1)
        _sift_loc_x, _sift_loc_y, _sift_loc_z = _sift_loc.nonzero(as_tuple=True)
        _sift_loc = _sift_loc[_sift_loc_x, _sift_loc_y, _sift_loc_z].reshape(-1, 5)  # todo 检查正确性

        _select_target = bbox_targets.reshape(-1, 5)[pos_index].unsqueeze(1).repeat(1, self.gmm.n_components,
                                                                                    1).reshape(-1, 5)
        _select_target_unique = _select_target.unique(dim=0)
        _select_target_unique_area = _select_target_unique[:, 2] * _select_target_unique[:, 3]  # compute area
        inside_flag_1 = torch.full((_select_target.shape[0], _select_target_unique.shape[0]), 0).to(_select_target)
        # inside_flag_2 = torch.full((_select_target.shape[0], _select_target.shape[0]), 0).to(_select_target)
        inside_flag_2 = torch.full((_select_target.shape[0], 1), 0).to(_select_target)
        pointsJf(_center_pos * stride[0], obb2poly(_select_target_unique, self.angle_version), inside_flag_1)
        # pointsJf(_center_pos * stride[0], obb2poly(_select_target), inside_flag_2)
        pointsJfAlign(_center_pos * stride[0], obb2poly(_select_target, self.angle_version), inside_flag_2)
        _candidate_index = inside_flag_1.sum(-1) > 1

        if _candidate_index.sum():
            select_inside_flag_1 = inside_flag_1[_candidate_index]
            select_area_matrix = select_inside_flag_1 * _select_target_unique_area.unsqueeze(0)
            select_area_matrix.masked_fill_(select_area_matrix == 0, float('inf'))
            seledt_values, select_indexes = torch.min(select_area_matrix, dim=1)
            inside_flag_1[_candidate_index] = 0
            inside_flag_1[_candidate_index, select_indexes] = 1

        _select_mask = (inside_flag_2.squeeze(-1) == 1) & (inside_flag_1.sum(-1) == 1)
        # _select_mask = (inside_flag_2.diagonal() == 1) & (inside_flag_1.sum(-1) == 1)
        _sift_index_value = _select_mask.nonzero(as_tuple=False).squeeze()
        _sift_score_pos = ((_select_mask * _score_pos).reshape(-1, self.gmm.n_components) / ((_select_mask * _score_pos).reshape(-1, self.gmm.n_components).sum(
            -1) + 1e-6).unsqueeze(-1)).reshape(-1)


        labels = labels.reshape(-1)  # [batch_size,num_anchors_in_a_lvl] --> [batch_size*num_anchors_in_a_lvl]
        label_weights = label_weights.reshape(-1)
        label_weights[pos_index] = 0

        _concat_label_weights = torch.cat((label_weights, _sift_score_pos))
        # _labels_ = torch.full_like(labels, self.cls_out_channels)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        _concat_cls_score = torch.cat((cls_score, _sift_value))
        # [batchszie, channel, H, W] --> [batchszie, H, W, channel] --> [., c]
        _concat_labels = torch.cat((labels, _label_pos))

        loss_cls = self.loss_cls(
            _concat_cls_score, _concat_labels, _concat_label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)  # [batchsize, num_bboxes, 5]
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        pt_pred = pt_pred.permute(0, 2, 3, 1).reshape(-1, self.point_num * 2)
        # [batch_size, 5*num_base_anchors, H, W] --> [batch_size, H, W, self.point_num * 2] --> [N, self.point_num * 2]

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 5)
            # 重新选择anchor
            _select_anchor = anchors[pos_index]
            _select_anchor = _select_anchor.unsqueeze(1).repeat(1, self.gmm.n_components, 1).reshape(-1, 5)
            _select_anchor = torch.cat((_center_pos * stride[0], _select_anchor[:, 2:]), dim=1)
            bbox_pred_t = self.bbox_coder.decode(_select_anchor, _sift_loc)

        if len(pos_index) > 0:
            _bbox_weights = bbox_weights.mean(dim=1)[pos_index]
            anchor_points = anchor_points.reshape(-1, self.point_num * 2)[pos_index]
            _pt_pred = (pt_pred[pos_index] * stride[0] + anchor_points) / 1024

            pt_targets = obb2poly(bbox_targets[pos_index], self.angle_version) / 1024
            loss_pt = self.loss_pt(
                _pt_pred,
                pt_targets,
                _bbox_weights)

            loss_bbox = self.loss_bbox(bbox_pred_t, _select_target,
                                       weight=_sift_score_pos.unsqueeze(-1).repeat(1, 5), avg_factor=num_total_samples)
        else:
            loss_pt = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0

        return loss_cls, loss_bbox, loss_pt

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'pt_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             pt_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        # cls：channel = num_classes * anchors; list[] five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        # box_pred: channel = anchors * 5;five tensors(depend on lvls) each is [bactchsize, channel, H, W]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        anchor_points_list, _ = self.get_anchor_points(anchor_list, self.point_num, device=anchor_list[0][0].device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # self.sampling主要根据sampling loss_cls的方式来决定
        # [[[], [],..(batch_size)], [[], [],..(batch_size)], ..(numlvls)]
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        concat_anchor_points_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_anchor_points_list.append(torch.cat(anchor_points_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        all_anchor_point_list = images_to_levels(concat_anchor_points_list,
                                                 num_level_anchors)
        # concat_anchor_list [[num_anchors,5],[num_anchors,5],...(batch_size)]
        # all_anchor_list [[batchsize, all_anchors_in_a_lvl, 5],[]...(numlvls)]
        loss_cls, loss_bbox, loss_pt = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            pt_preds,
            all_anchor_list,
            all_anchor_point_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            self.anchor_generator.strides,
            num_total_samples=num_total_samples)
        # avg_factor = sum(item.data.sum() for item in avg_num).float()
        # _avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        # loss_bbox_ = list(map(lambda x: x / _avg_factor, loss_bbox))
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_pt=loss_pt)

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            num_level_anchors,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        inside_flags = ranchor_inside_flags(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            self.train_cfg.allowed_border)
        # check whether the anchor inside the border
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(
            anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        # TODO: godeeper

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # 由于产生的一些不合格的anchor在该方法的开头已经被去掉，为了保证anchor的总数所以要在对应的结果后加上0
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        # e.t.c [147456, 36864, 9216, 2304, 576]
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_anchor_point_list = []
        concat_valid_flag_list = []

        # main function is to concat all lvls anchor in one images
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        #  [[[],[],[],..(num_lvls)],[[],[],..(numlvls)],..(batch_size)] --> [[],[],..(batch_size)]
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        # [[[], [],..(batch_size)], [[], [],..(batch_size)], ..(numlvls)]
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list,)
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):

        # 这里的cfg时proposal cfg
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, stride in zip(cls_score_list, bbox_pred_list, mlvl_anchors,
                                                         self.anchor_generator.strides):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[:, :4] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # class_agnostic = cfg.get('class_agnostic', True)
        if with_nms:
            det_bboxes, det_labels = multiclass_nms_r(mlvl_bboxes, mlvl_scores,
                                                      cfg.score_thr, cfg.nms,
                                                      cfg.max_per_img)

            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        num_imgs = len(cls_scores[0])

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]
        mlvl_anchor_points, _ = self.get_anchor_points(anchor_list, self.point_num, device=device)
        # [[anchors_num, 5],..(num_lvls)]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:

                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors,
                                                    img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    mlvl_anchors,
                                                    img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            # proposals [n, 5]
            result_list.append(proposals)
        return result_list

    # modified from https://github.com/FangyunWei/PointSetAnchor
    def generate_anchor_points(self,
                               anchors,
                               anchor_points_num,
                               device):
        """

        :param anchors: anchors of single level, Tensor(n, 4), note: shape(x, y, w, h, angle)
        :param anchor_points_num: template point number of each level
        :param device: device
        :return: generated template points,
                Tensor(n, anchor_points_num x 2)
        """
        n = anchors.shape[0]
        anchor_points = torch.zeros((n, anchor_points_num, 2),
                                    device=device)
        # template point number in each side
        anchor_points_count_in_line = anchor_points.new_zeros((n, 4), dtype=torch.long)
        current_anchors = anchors.clone()
        center_x = current_anchors[:, 0]
        center_y = current_anchors[:, 1]
        current_anchors_w = current_anchors[:, 2]
        current_anchors_h = current_anchors[:, 3]
        anchors_x1 = center_x - (current_anchors_w - 1) / 2  # top left
        anchors_x2 = center_x + (current_anchors_w - 1) / 2  # bottom right
        anchors_y1 = center_y - (current_anchors_h - 1) / 2  # top left
        anchors_y2 = center_y + (current_anchors_h - 1) / 2  # bottom right
        anchors_w = anchors_x2 - anchors_x1 + 1
        anchors_h = anchors_y2 - anchors_y1 + 1  # single lvl point num (n)
        # total template points num =
        # num_w_side*2 + num_extra + num_h_side*2
        num_wh_side = anchor_points_num // 2
        num_extra = anchor_points_num % 2
        unique_ws = anchors_w.unique(sorted=False)
        for unique_w in unique_ws:
            idx = (anchors_w == unique_w).nonzero(as_tuple=False).squeeze(1)
            current_x1 = anchors_x1[idx]
            current_x2 = anchors_x2[idx]
            current_y1 = anchors_y1[idx]
            current_y2 = anchors_y2[idx]
            current_w = anchors_w[idx[0]].item()
            current_h = anchors_h[idx[0]].item()
            current_wh = current_w + current_h
            # template points num for each w side
            num_w_side = int(current_w / current_wh * num_wh_side)
            # template points num for each h side
            num_h_side = num_wh_side - num_w_side
            # template points on top side
            top_invervals = torch.linspace(0, current_w - 1, num_w_side + num_extra + 2, device=device)[1:-1]
            top_x = current_x1.repeat(len(top_invervals), 1).permute(1, 0) + \
                    top_invervals.repeat(len(idx), 1)  # shape (n, num_w_side)
            top_y = current_y1.repeat(len(top_invervals), 1).permute(1, 0)
            top_points = torch.stack([top_x, top_y], dim=2)
            # template points on right side
            right_invervals = torch.linspace(0, current_h - 1, num_h_side + 2, device=device)[1:-1]
            right_x = current_x2.repeat(len(right_invervals), 1).permute(1, 0)
            right_y = current_y1.repeat(len(right_invervals), 1).permute(1, 0) + \
                      right_invervals.repeat(len(idx), 1)
            right_points = torch.stack([right_x, right_y], dim=2)
            # template points on bottom side
            bottom_invervals = torch.linspace(current_w - 1, 0, num_w_side + 2, device=device)[1:-1]
            bottom_x = current_x1.repeat(len(bottom_invervals), 1).permute(1, 0) + \
                       bottom_invervals.repeat(len(idx), 1)
            bottom_y = current_y2.repeat(len(bottom_invervals), 1).permute(1, 0)
            bottom_points = torch.stack([bottom_x, bottom_y], dim=2)
            # template points on left side
            left_invervals = torch.linspace(current_h - 1, 0, num_h_side + 2, device=device)[1:-1]
            left_x = current_x1.repeat(len(left_invervals), 1).permute(1, 0)
            left_y = current_y1.repeat(len(left_invervals), 1).permute(1, 0) + \
                     left_invervals.repeat(len(idx), 1)
            left_points = torch.stack([left_x, left_y], dim=2)
            # template points should be in order
            all_points = torch.cat((top_points, right_points, bottom_points, left_points), dim=1)
            anchor_points[idx, :, 0: 2] = all_points
            anchor_points_count_in_line[idx, :] = anchor_points_count_in_line.new_tensor([num_w_side + num_extra,
                                                                                          num_h_side,
                                                                                          num_w_side,
                                                                                          num_h_side])
        t_n, t_tn, t_lv = anchor_points.shape
        anchor_points = anchor_points.view(t_n, t_tn * t_lv)
        return anchor_points, anchor_points_count_in_line

    def get_anchor_points(self, anchor_list,
                          anchor_points_num,
                          device='cuda'):
        """

        :param anchor_list: different anchor bbxs of all images
        :param anchor_points_num: total template point number
        :param device: device
        :return: generated anchor points for all images
        (x, y)
        """
        num_imgs = len(anchor_list)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        anchors = anchor_list[0]
        num_levels = len(anchors)
        multi_level_anchor_points = []
        multi_level_anchor_points_count = []
        for i in range(num_levels):
            points, points_count_in_line = self.generate_anchor_points(anchors[i].clone(),
                                                                       anchor_points_num,
                                                                       device)
            multi_level_anchor_points.append(points)
            multi_level_anchor_points_count.append(points_count_in_line)
        anchor_points_list = [multi_level_anchor_points for _ in range(num_imgs)]
        anchor_points_count_list = [multi_level_anchor_points_count for _ in range(num_imgs)]
        return anchor_points_list, anchor_points_count_list

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside