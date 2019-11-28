import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import remove_small_boxes
from fcos_core.structures.boxlist_ops import boxes_to_masks
from fcos_core.layers.misc import interpolate
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
import math

INF = 100000000

class EmbedMaskPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        cfg
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(EmbedMaskPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        self.mask_scale_factor = cfg.MODEL.EMBED_MASK.MASK_SCALE_FACTOR
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        self.mask_th = cfg.MODEL.EMBED_MASK.MASK_TH
        self.post_process_masks = cfg.MODEL.EMBED_MASK.POSTPROCESS_MASKS

        self.object_sizes_of_interest = [
            [-1, cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[0]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[0], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[1]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[1], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[2]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[2], cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[3]],
            [cfg.MODEL.EMBED_MASK.FPN_INTEREST_SPLIT[3], INF]
        ]

        self.fix_margin = cfg.MODEL.EMBED_MASK.FIX_MARGIN
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.init_margin = -math.log(0.5) / (prior_margin ** 2)

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            proposal_embed, proposal_margin, image_sizes, level):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        proposal_embed = proposal_embed.view(N, -1, H, W).permute(0, 2, 3, 1)
        proposal_embed = proposal_embed.reshape(N, H*W, -1)
        proposal_margin = proposal_margin.view(N, 1, H, W).permute(0, 2, 3, 1)
        proposal_margin = proposal_margin.reshape(N, -1)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_proposal_embed = proposal_embed[i]
            per_proposal_embed = per_proposal_embed[per_box_loc]
            per_proposal_margin = proposal_margin[i][per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_proposal_embed = per_proposal_embed[top_k_indices]
                per_proposal_margin = per_proposal_margin[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("proposal_embed", per_proposal_embed)
            boxlist.add_field("proposal_margin", per_proposal_margin)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def compute_mask_prob(self, pixel_embed, proposal_embed, proposal_margin):
        m_h, m_w = pixel_embed.shape[-2:]
        obj_num = proposal_embed.shape[0]
        pixel_embed = pixel_embed.permute(1, 2, 0).unsqueeze(0).expand(obj_num, -1, -1, -1)
        proposal_embed = proposal_embed.view(obj_num, 1, 1, -1).expand(-1, m_h, m_w, -1)
        proposal_margin = proposal_margin.view(obj_num, 1, 1).expand(-1, m_h, m_w)
        mask_var = torch.sum((pixel_embed - proposal_embed) ** 2, dim=3)
        mask_prob = torch.exp(-mask_var*proposal_margin)

        return mask_prob

    def forward_for_mask(self, boxlists, pixel_embed):
        N, dim, m_h, m_w = pixel_embed.shape
        new_boxlists = []
        stride = self.fpn_strides[0] / self.mask_scale_factor
        for im in range(N):
            boxlist = boxlists[im]
            boxes = boxlist.bbox
            input_w, input_h = boxlist.size
            proposal_embed = boxlist.get_field('proposal_embed')
            if len(proposal_embed) == 0:
                new_boxlist = BoxList(boxes, boxlist.size, mode="xyxy")
                new_boxlist.add_field("labels", boxlist.get_field("labels"))
                new_boxlist.add_field("scores", boxlist.get_field("scores"))
                new_boxlist.add_field('mask', torch.tensor([]))
                if self.post_process_masks:
                    new_boxlist.add_field('stride', torch.tensor(1))
                    new_boxlist.add_field('mask_th', torch.tensor(0.0))
                else:
                    new_boxlist.add_field('stride', torch.tensor(stride))
                    new_boxlist.add_field('mask_th', torch.tensor(self.mask_th))

                new_boxlists.append(new_boxlist)
                continue

            mask_boxes = boxes / stride
            box_masks = boxes_to_masks(mask_boxes, m_h, m_w)
            proposal_margin = boxlist.get_field('proposal_margin')
            mask_prob = self.compute_mask_prob(pixel_embed[im], proposal_embed, proposal_margin)
            masks = mask_prob * box_masks.float()

            if self.post_process_masks:
                masks = torch.nn.functional.interpolate(input=masks.unsqueeze(1).float(), scale_factor=stride,
                                                        mode="bilinear", align_corners=False).gt(self.mask_th)
                masks = masks[:, 0, :input_h, :input_w]

            new_boxlist = BoxList(boxes, boxlist.size, mode="xyxy")
            new_boxlist.add_field('mask', masks)
            new_boxlist.add_field("labels", boxlist.get_field("labels"))
            new_boxlist.add_field("scores", boxlist.get_field("scores"))
            if self.post_process_masks:
                new_boxlist.add_field('stride', torch.tensor(1))
                new_boxlist.add_field('mask_th', torch.tensor(0.0))
            else:
                new_boxlist.add_field('stride', torch.tensor(stride))
                new_boxlist.add_field('mask_th', torch.tensor(self.mask_th))

            new_boxlists.append(new_boxlist)

        return new_boxlists

    def forward(self, locations, box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed, image_sizes, targets):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        for i, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            em = proposal_embed[i]
            mar = proposal_margin[i]
            if self.fix_margin:
                mar = torch.ones_like(mar) * self.init_margin
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, em, mar, image_sizes, i
                )
            )
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        # resize pixel embedding for higher resolution
        N, dim, m_h, m_w = pixel_embed.shape
        o_h = m_h * self.mask_scale_factor
        o_w = m_w * self.mask_scale_factor
        pixel_embed = interpolate(pixel_embed, size=(o_h, o_w), mode='bilinear', align_corners=False)

        boxlists = self.forward_for_mask(boxlists, pixel_embed)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_embed_mask_postprocessor(config):
    pre_nms_thresh = config.MODEL.EMBED_MASK.INFERENCE_TH
    pre_nms_top_n = config.MODEL.EMBED_MASK.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.EMBED_MASK.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG

    box_selector = EmbedMaskPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.EMBED_MASK.NUM_CLASSES,
        cfg=config
    )

    return box_selector
