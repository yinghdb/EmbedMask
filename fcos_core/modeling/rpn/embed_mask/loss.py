"""
This file contains specific functions for computing losses of FCOS
file
"""
import os
import torch
from torch import nn

from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import LovaszHinge
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import boxlist_overlap
from fcos_core.structures.boxlist_ops import crop_by_box
from fcos_core.structures.boxlist_ops import boxes_to_masks
from fcos_core.layers.misc import interpolate
from fcos_core.structures.bounding_box import BoxList

import math

INF = 100000000

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

class EmbedMaskLossComputation(object):
    """
    This class computes the FCOS losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.EMBED_MASK.LOSS_GAMMA,
            cfg.MODEL.EMBED_MASK.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        self.center_on = cfg.MODEL.EMBED_MASK.CENTER_ON
        self.center_sampling_radius = cfg.MODEL.EMBED_MASK.CENTER_POS_RADIOS
        self.iou_loss_type = cfg.MODEL.EMBED_MASK.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.EMBED_MASK.NORM_REG_TARGETS

        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

        ########## mask prediction ############
        self.mask_loss_func = LovaszHinge(reduction='none')

        self.mask_scale_factor = cfg.MODEL.EMBED_MASK.MASK_SCALE_FACTOR
        self.object_sizes_of_interest = [
            [-1, cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[0]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[0], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[1]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[1], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[2]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[2], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[3]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[3], INF]
        ]
        self.sample_in_mask = cfg.MODEL.EMBED_MASK.SAMPLE_IN_MASK
        self.sample_pos_iou_th = cfg.MODEL.EMBED_MASK.SAMPLE_POS_IOU_TH

        self.box_padding = cfg.MODEL.EMBED_MASK.BOX_PADDING

        self.fix_margin = cfg.MODEL.EMBED_MASK.FIX_MARGIN
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.init_margin = -math.log(0.5) / (prior_margin ** 2)

        self.loss_mask_alpha = cfg.MODEL.EMBED_MASK.LOSS_MASK_ALPHA
        self.loss_smooth_alpha = cfg.MODEL.EMBED_MASK.LOSS_SMOOTH_ALPHA

    def fresh_alphas(self, cfg):
        self.loss_mask_alpha = cfg.MODEL.EMBED_MASK.LOSS_MASK_ALPHA
        self.loss_smooth_alpha = cfg.MODEL.EMBED_MASK.LOSS_SMOOTH_ALPHA

    def prepare_targets(self, points, targets, im_w, im_h):
        object_sizes_of_interest = self.object_sizes_of_interest
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, matched_idxes = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, im_w, im_h
        )

        labels_split = []
        reg_targets_split = []
        for i in range(len(labels)):
            labels_split.append(torch.split(labels[i], num_points_per_level, dim=0))
            reg_targets_split.append(torch.split(reg_targets[i], num_points_per_level, dim=0))

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels_split], dim=0)
            )
            reg_targets_per_level = \
                torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets_split], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        matched_idxes = torch.stack(matched_idxes)

        return labels_level_first, reg_targets_level_first, labels, reg_targets, matched_idxes

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest, im_w, im_h):
        labels = []
        reg_targets = []
        matched_idxes = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            if self.center_on:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs,
                    ys,
                    radius=self.center_sampling_radius)
            else:
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            if self.sample_in_mask:
                masks_ori = targets_per_im.get_field('masks').convert('mask').instances.masks.to(device=locations.device)  # n, h, w
                masks_ori = masks_ori.permute(1, 2, 0) # h, w, n
                masks = masks_ori.new_zeros((im_h, im_w, masks_ori.shape[2]))
                masks[:masks_ori.shape[0], :masks_ori.shape[1], :] = masks_ori
                mask_targets = masks[ys.long(), xs.long()]
                locations_to_gt_area[mask_targets == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_aera == INF] = 0
            locations_to_gt_inds[locations_to_min_aera == INF] = -1

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            matched_idxes.append(locations_to_gt_inds)

        return labels, reg_targets, matched_idxes

    def get_pos_proposal_indexes(self, locations, box_regression, matched_idxes, targets):
        locations = torch.cat(locations, dim=0)
        pos_indexes_for_targets = []
        for im in range(len(targets)):
            pos_indexes_for_targets_per_im = []
            box_regression_im = [box_regression[l][im].detach().view(4, -1).transpose(0, 1).contiguous() * self.fpn_strides[l] for l in
                                 range(len(box_regression))]
            box_regression_im = torch.cat(box_regression_im, dim=0)
            for t_id in range(len(targets[im])):
                valid = matched_idxes[im] == t_id
                if valid.sum() == 0:
                    pos_indexes_for_targets_per_im.append(valid.new_tensor([]))
                    continue
                valid_location = locations[valid]
                valid_regression = box_regression_im[valid]
                detections = torch.stack([
                    valid_location[:, 0] - valid_regression[:, 0],
                    valid_location[:, 1] - valid_regression[:, 1],
                    valid_location[:, 0] + valid_regression[:, 2],
                    valid_location[:, 1] + valid_regression[:, 3],
                ], dim=1)
                detect_boxlist = BoxList(detections, targets[im].size, mode="xyxy")
                target_boxlist = BoxList(targets[im].bbox[t_id:t_id+1], targets[im].size, mode="xyxy")
                match_quality_matrix = boxlist_iou(detect_boxlist, target_boxlist)

                pos_labels_per_target = torch.zeros_like(valid)
                iou_in_target = match_quality_matrix[:, 0]
                if iou_in_target.max() > self.sample_pos_iou_th:
                    pos_in_target = (iou_in_target > self.sample_pos_iou_th)
                else:
                    pos_in_target = (iou_in_target == iou_in_target.max())
                pos_labels_per_target[valid] = pos_in_target

                pos_indexes_for_targets_per_im.append(pos_labels_per_target.nonzero().squeeze(1))
            pos_indexes_for_targets.append(pos_indexes_for_targets_per_im)

        return pos_indexes_for_targets

    def get_proposal_element(self, features, poses):
        N, dim = features[0].shape[:2]
        features_flatten = torch.cat(
            [features_per_level.view(N, dim, -1) for features_per_level in features], dim=2
        ).transpose(1, 2).contiguous()
        pos_features_for_targets = []
        for im in range(N):
            pos_features_for_targets_im = []
            for t_id in range(len(poses[im])):
                if len(poses[im][t_id]) == 0:
                    pos_features_for_targets_im.append(features_flatten.new_tensor([]))
                else:
                    pos_features_for_targets_im.append(features_flatten[im][poses[im][t_id]])
            pos_features_for_targets.append(pos_features_for_targets_im)
        return pos_features_for_targets

    def calculate_means(self, features):
        means = []
        for im in range(len(features)):
            means_im = []
            for t_id in range(len(features[im])):
                if len(features[im][t_id]) == 0:
                    means_im.append(features[im][t_id])
                else:
                    means_im.append(features[im][t_id].mean(dim=0).unsqueeze(0))
            means.append(means_im)
        return means

    def prepare_masks(self, o_h, o_w, r_h, r_w, targets_masks):
        masks = []
        for im_i in range(len(targets_masks)):
            mask_t = targets_masks[im_i]
            if len(mask_t) == 0:
                masks.append(mask_t.new_tensor([]))
                continue
            n, h, w = mask_t.shape
            mask = mask_t.new_zeros((n, r_h, r_w))
            mask[:, :h, :w] = mask_t
            resized_mask = interpolate(
                input=mask.float().unsqueeze(0), size=(o_h, o_w), mode="bilinear", align_corners=False,
            )[0].gt(0)

            masks.append(resized_mask)

        return masks

    def compute_mask_prob(self, proposal_embed, proposal_margin, pixel_embed):
        m_h, m_w = pixel_embed.shape[-2:]
        obj_num = proposal_embed.shape[0]
        pixel_embed = pixel_embed.permute(1, 2, 0).unsqueeze(0).expand(obj_num, -1, -1, -1)
        proposal_embed = proposal_embed.view(obj_num, 1, 1, -1).expand(-1, m_h, m_w, -1)
        if self.fix_margin:
            proposal_margin = proposal_margin.new_ones(obj_num, m_h, m_w) * self.init_margin
        else:
            proposal_margin = proposal_margin.view(obj_num, 1, 1).expand(-1, m_h, m_w)
        mask_var = torch.sum((pixel_embed - proposal_embed) ** 2, dim=3)
        mask_prob = torch.exp(-mask_var*proposal_margin)

        return mask_prob

    def __call__(self, locations, box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        num_classes = box_cls[0].size(1)
        im_h = box_cls[4].shape[2] * self.fpn_strides[4]
        im_w = box_cls[4].shape[3] * self.fpn_strides[4]
        labels_per_level, reg_targets_per_level, labels, reg_targets, matched_idxes = self.prepare_targets(locations, targets, im_w, im_h)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels_per_level)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels_per_level[l].reshape(-1))
            reg_targets_flatten.append(reg_targets_per_level[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu


        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        #################################### Mask Related Losses ######################################
        # get positive proposal labels for each gt instance
        pos_proposal_labels_for_targets = self.get_pos_proposal_indexes(locations, box_regression, matched_idxes, targets)

        # get positive samples of embeddings & margins for each gt instance
        proposal_embed_for_targets = self.get_proposal_element(proposal_embed, pos_proposal_labels_for_targets)
        proposal_margin_for_targets = self.get_proposal_element(proposal_margin, pos_proposal_labels_for_targets)

        # get proposal embedding & margin means
        embedding_means = self.calculate_means(proposal_embed_for_targets)
        margin_means = self.calculate_means(proposal_margin_for_targets)

        ############ SMOOTH_LOSS ##############
        smooth_loss = box_cls[0].new_tensor(0.0)
        if self.loss_smooth_alpha > 0:
            N = len(proposal_embed_for_targets)
            for im in range(N):
                target_num = len(proposal_embed_for_targets[im])
                smooth_loss_im = box_cls[0].new_tensor(0.0)
                for t_id in range(target_num):
                    if len(embedding_means[im][t_id])>0:
                        smooth_loss_im += torch.sum((proposal_embed_for_targets[im][t_id]-embedding_means[im][t_id])**2) + \
                            torch.sum((proposal_margin_for_targets[im][t_id] - margin_means[im][t_id]) ** 2)
                if target_num > 0:
                    smooth_loss += (smooth_loss_im / target_num)
            smooth_loss = smooth_loss / N * self.loss_smooth_alpha
        ######## MEANINGLESS_LOSS #######
        for i in range(len(proposal_embed)):
            smooth_loss += 0 * proposal_embed[i].sum()
            smooth_loss += 0 * proposal_margin[i].sum()
        smooth_loss += 0 * pixel_embed.sum()
        ############ Mask Losses ##############
        # get target masks in prefer size
        N, _, m_h, m_w = pixel_embed.shape
        o_h = m_h * self.mask_scale_factor
        o_w = m_w * self.mask_scale_factor
        r_h = int(m_h * self.fpn_strides[0])
        r_w = int(m_w * self.fpn_strides[0])
        stride = self.fpn_strides[0] / self.mask_scale_factor
        targets_masks = [target_im.get_field('masks').convert('mask').instances.masks.to(device=pixel_embed.device) for target_im in targets]
        masks_t = self.prepare_masks(o_h, o_w, r_h, r_w, targets_masks)
        pixel_embed = interpolate(input=pixel_embed, size=(o_h, o_w), mode="bilinear", align_corners=False)

        mask_loss = box_cls[0].new_tensor(0.0)
        proposal_embed_samples = embedding_means
        proposal_margin_samples = margin_means
        if self.loss_mask_alpha > 0:
            for im in range(N):
                mask_loss_im = box_cls[0].new_tensor(0.0)
                target_num = len(proposal_embed_for_targets[im])
                for t_id in range(target_num):
                    if len(proposal_embed_samples[im][t_id]) == 0:
                        continue
                    masks_prob = self.compute_mask_prob(proposal_embed_samples[im][t_id],
                                                        proposal_margin_samples[im][t_id],
                                                        pixel_embed[im])
                    sample_num = len(masks_prob)
                    masks_t_id = masks_t[im][t_id]
                    boxes_t_id = targets[im].bbox[t_id] / stride
                    masks_prob_crop, crop_mask = crop_by_box(masks_prob, boxes_t_id, self.box_padding)
                    mask_loss_per_target = self.mask_loss_func(masks_prob_crop, masks_t_id.unsqueeze(0).expand(sample_num, -1, -1).float(),
                                                               mask=crop_mask, act=True)

                    mask_loss_im += mask_loss_per_target.mean()
                if target_num > 0:
                    mask_loss += mask_loss_im / target_num
            mask_loss = mask_loss / N * self.loss_mask_alpha

        return cls_loss, reg_loss, centerness_loss, mask_loss, smooth_loss

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

def make_embed_mask_loss_evaluator(cfg):
    loss_evaluator = EmbedMaskLossComputation(cfg)
    return loss_evaluator
