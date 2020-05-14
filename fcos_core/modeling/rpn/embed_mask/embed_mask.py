import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_embed_mask_postprocessor
from .loss import make_embed_mask_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d

class EmbedMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(EmbedMaskHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.EMBED_MASK.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.EMBED_MASK.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.EMBED_MASK.USE_DCN_IN_TOWER

        num_classes = cfg.MODEL.EMBED_MASK.NUM_CLASSES - 1
        embed_dim = cfg.MODEL.EMBED_MASK.EMBED_DIM
        prior_margin = cfg.MODEL.EMBED_MASK.PRIOR_MARGIN
        self.init_sigma_bias = math.log(-math.log(0.5) / (prior_margin ** 2))

        cls_tower = []
        bbox_tower = []
        mask_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
            mask_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            mask_tower.append(nn.GroupNorm(32, in_channels))
            mask_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.EMBED_MASK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        ########### Mask Predictions ############
        # proposal embedding
        self.proposal_spatial_embed_pred = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.proposal_other_embed_pred = nn.Conv2d(
            in_channels, embed_dim-2, kernel_size=3, stride=1, padding=1, bias=True
        )
        for modules in [self.proposal_spatial_embed_pred, self.proposal_other_embed_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # proposal margin
        self.proposal_margin_pred = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True
        )
        torch.nn.init.normal_(self.proposal_margin_pred.weight, std=0.01)
        torch.nn.init.constant_(self.proposal_margin_pred.bias, self.init_sigma_bias)

        # pixel embedding
        self.add_module('mask_tower', nn.Sequential(*mask_tower))
        self.pixel_spatial_embed_pred = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.pixel_other_embed_pred = nn.Conv2d(
            in_channels, embed_dim-2, kernel_size=3, stride=1, padding=1, bias=True
        )
        for modules in [self.mask_tower, self.pixel_spatial_embed_pred, self.pixel_other_embed_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.position_scale = Scale(init_value=1.0)
        
    def forward(self, x, locations, benchmark=False, timers=None):
        logits = []
        bbox_reg = []
        centerness = []
        proposal_margin = []
        proposal_embed = []
        if benchmark and timers is not None:
            torch.cuda.synchronize()
            timers[2].tic()
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

            # ############### Mask Prediction ###########
            embed_x = box_tower

            h, w = embed_x.size()[-2:]
            proposal_spatial_embd = self.proposal_spatial_embed_pred(embed_x)
            proposal_other_embd = self.proposal_other_embed_pred(embed_x)
            coordinates = locations[l].transpose(0, 1).contiguous().view(2, h, w).unsqueeze(0).expand(embed_x.shape[0], -1, -1, -1)
            scaled_coordinates = self.position_scale(coordinates) / 100.0
            proposal_spatial_embd = scaled_coordinates + proposal_spatial_embd
            proposal_embed.append(torch.cat([proposal_spatial_embd, proposal_other_embd], dim=1))

            margin_x = box_tower / 32
            proposal_margin.append(torch.exp(self.proposal_margin_pred(margin_x)))

        if benchmark and timers is not None:
            torch.cuda.synchronize()
            timers[2].toc()
            timers[3].tic()
        # pixel embedding
        mask_x = x[0]
        mask_x = self.mask_tower(mask_x)

        h, w = mask_x.size()[-2:]
        pixel_spatial_embd = self.pixel_spatial_embed_pred(mask_x)
        pixel_other_embd = self.pixel_other_embed_pred(mask_x)
        coordinates = locations[0].transpose(0, 1).contiguous().view(2, h, w).unsqueeze(0).expand(mask_x.shape[0], -1, -1, -1)
        scaled_coordinates = self.position_scale(coordinates) / 100.0
        pixel_spatial_embd = scaled_coordinates + pixel_spatial_embd
        pixel_embed = torch.cat([pixel_spatial_embd, pixel_other_embd], dim=1)

        if benchmark and timers is not None:
            torch.cuda.synchronize()
            timers[3].toc()

        return logits, bbox_reg, centerness, proposal_embed, proposal_margin, pixel_embed

class EmbedMaskModule(torch.nn.Module):
    """
    Module for EmbedMask computation. Takes feature maps from the backbone and
    EmbedMask outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(EmbedMaskModule, self).__init__()

        head = EmbedMaskHead(cfg, in_channels)

        box_selector_test = make_embed_mask_postprocessor(cfg)
        loss_evaluator = make_embed_mask_loss_evaluator(cfg)

        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.EMBED_MASK.FPN_STRIDES

    def fresh_alpha(self, cfg):
        self.loss_evaluator.fresh_alphas(cfg)

    def forward(self, images, features, targets=None, benchmark=False, timers=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        if benchmark and timers is not None:
            torch.cuda.synchronize()
            timers[1].tic()
        locations = self.compute_locations(features)
        if benchmark and timers is not None:
            torch.cuda.synchronize()
            timers[1].toc()
        box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed = self.head(features, locations, benchmark, timers)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness,
                proposal_embed, proposal_margin, pixel_embed,
                targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, proposal_embed, proposal_margin, pixel_embed,
                images.image_sizes, targets, benchmark, timers
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed, targets):
        loss_box_cls, loss_box_reg, loss_centerness, \
        mask_loss = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, proposal_embed, proposal_margin, pixel_embed, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            'mask_loss': mask_loss,
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness,
                      proposal_embed, proposal_margin, pixel_embed,
                      image_sizes, targets, benchmark=False, timers=None):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, proposal_embed, proposal_margin, pixel_embed,
            image_sizes, targets, benchmark, timers
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_embed_mask(cfg, in_channels):
    return EmbedMaskModule(cfg, in_channels)
