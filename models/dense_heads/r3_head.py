# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmdet.models.utils import select_single_mlvl
from mmdet.utils import InstanceList, OptInstanceList

from mmengine.config import ConfigDict
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from .rotated_retina_head import RotatedRetinaHead
from mmdet.models.task_modules.prior_generators import (AnchorGenerator,
                                                        anchor_inside_flags)

from mmdet.registry import TASK_UTILS
from mmdet.utils import (reduce_mean)
from mmengine.structures import InstanceData
from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmdet.structures.bbox import cat_boxes


@MODELS.register_module()
class R3Head(RotatedRetinaHead):
    r"""An anchor-based head used in `R3Det
    <https://arxiv.org/pdf/1908.05612.pdf>`_.
    """  # noqa: W605

    def filter_bboxes(self, cls_scores: List[Tensor],
                      bbox_preds: List[Tensor]) -> List[List[Tensor]]:
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level
            of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5).tensor

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(
                    RotatedBoxes(best_anchor_i), best_pred_i)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list


@MODELS.register_module()
class R3RefineHead(RotatedRetinaHead):
    r"""An anchor-based head used in `R3Det
    <https://arxiv.org/pdf/1908.05612.pdf>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        frm_cfg (dict): Config of the feature refine module.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 frm_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)
        if self.train_cfg:
            self.cls_assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.reg_assigner = TASK_UTILS.build(self.train_cfg['reg_assigner'])

        self.feat_refine_module = MODELS.build(frm_cfg)
        self.bboxes_as_anchors = None

        # we use the global list in loss
        self.cls_num_pos_samples_per_level = [
            0. for _ in range(len(self.prior_generator.strides))
        ]
        self.reg_num_pos_samples_per_level = [
            0. for _ in range(len(self.prior_generator.strides))
        ]
    # def loss_by_feat(self,
    #                  cls_scores: List[Tensor],
    #                  bbox_preds: List[Tensor],
    #                  batch_gt_instances: InstanceList,
    #                  batch_img_metas: List[dict],
    #                  batch_gt_instances_ignore: OptInstanceList = None,
    #                  rois: List[Tensor] = None) -> dict:
    #     """Calculate the loss based on the features extracted by the detection
    #     head.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             has shape (N, num_anchors * num_classes, H, W).
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W).
    #         batch_gt_instances (list[:obj:`InstanceData`]): Batch of
    #             gt_instance. It usually includes ``bboxes`` and ``labels``
    #             attributes.
    #         batch_img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
    #             Batch of gt_instances_ignore. It includes ``bboxes`` attribute
    #             data that is ignored during training and testing.
    #             Defaults to None.
    #         rois (list[Tensor])
    #
    #     Returns:
    #         dict: A dictionary of loss components.
    #     """
    #     assert rois is not None
    #     self.bboxes_as_anchors = rois
    #     return super(RotatedRetinaHead, self).loss_by_feat(
    #         cls_scores=cls_scores,
    #         bbox_preds=bbox_preds,
    #         batch_gt_instances=batch_gt_instances,
    #         batch_img_metas=batch_img_metas,
    #         batch_gt_instances_ignore=batch_gt_instances_ignore)

    def calc_reweight_factor(self, labels_list):
        """Compute reweight_factor for regression and classification loss."""
        # get pos samples for each level
        bg_class_ind = self.num_classes
        for ii, each_level_label in enumerate(labels_list):
            pos_inds = ((each_level_label >= 0) &
                        (each_level_label < bg_class_ind)).nonzero(
                            as_tuple=False).squeeze(1)
            self.cls_num_pos_samples_per_level[ii] += len(pos_inds)
        # get reweight factor from 1 ~ 2 with bilinear interpolation
        min_pos_samples = min(self.cls_num_pos_samples_per_level)
        max_pos_samples = max(self.cls_num_pos_samples_per_level)
        interval = 1. / (max_pos_samples - min_pos_samples + 1e-10)
        reweight_factor_per_level = []
        for pos_samples in self.cls_num_pos_samples_per_level:
            factor = 2. - (pos_samples - min_pos_samples) * interval
            reweight_factor_per_level.append(factor)
        return reweight_factor_per_level

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None,
            rois: List[Tensor] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        assert rois is not None
        self.bboxes_as_anchors = rois

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # calculate common vars for cls and reg assigners at once
        targets_com = self.process_predictions_and_anchors(
            anchor_list, valid_flag_list, cls_scores, bbox_preds, batch_img_metas,
            batch_gt_instances_ignore)
        (anchor_list, valid_flag_list, num_level_anchors_list, cls_score_list,
         bbox_pred_list, gt_bboxes_ignore_list) = targets_com

        gt_bboxes = batch_gt_instances[0].bboxes
        gt_labels = batch_gt_instances[0].labels
        # cls_targets = self.get_cls_targets(
        #     anchor_list,
        #     valid_flag_list,
        #     batch_gt_instances,
        #     batch_img_metas,
        #     batch_gt_instances_ignore=batch_gt_instances_ignore,
        #     label_channels=label_channels)

        # classification branch assigner
        # get_cls_targets
        cls_targets = self.get_cls_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            gt_bboxes,
            batch_img_metas,
            batch_gt_instances,
            gt_bboxes_ignore_list=gt_bboxes_ignore_list,
            gt_labels_list=gt_labels,
            label_channels=label_channels
            )

        # (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        #  avg_factor) = cls_targets
        if cls_targets is None:
            return None

        (cls_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        # loss_cls_single
        cls_losses_cls, = multi_apply(
            self.loss_cls_single,
            cls_scores,
            labels_list,
            label_weights_list,
            reweight_factor_per_level,
            num_total_samples=num_total_samples)

        # regression branch assigner
        reg_targets = self.get_reg_targets(
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            cls_score_list,
            bbox_pred_list,
            # gt_bboxes,
            batch_img_metas,
            batch_gt_instances,
            gt_bboxes_ignore_list=gt_bboxes_ignore_list,
            # gt_labels_list=gt_labels,
            label_channels=label_channels)
        if reg_targets is None:
            return None

        (reg_anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        reweight_factor_per_level = self.calc_reweight_factor(labels_list)
        # loss_reg_single
        reg_losses_bbox, = multi_apply(
            self.loss_reg_single,
            reg_anchor_list,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            reweight_factor_per_level,
            num_total_samples=num_total_samples)

        # # anchor number of multi levels
        # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # # concat all level anchors and flags to a single tensor
        # concat_anchor_list = []
        # for i in range(len(anchor_list)):
        #     concat_anchor_list.append(cat_boxes(anchor_list[i]))
        # all_anchor_list = images_to_levels(concat_anchor_list,
        #                                    num_level_anchors)

        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_by_feat_single,
        #     cls_scores,
        #     bbox_preds,
        #     all_anchor_list,
        #     labels_list,
        #     label_weights_list,
        #     bbox_targets_list,
        #     bbox_weights_list,
        #     avg_factor=avg_factor)
        return dict(loss_cls=cls_losses_cls, loss_bbox=reg_losses_bbox)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        anchor_list = [[
            RotatedBoxes(bboxes_img_lvl).detach()
            for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 5, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=rois[img_id],
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def loss_cls_single(self, cls_score, labels, label_weights,
                        reweight_factor, num_total_samples):
        """Compute cls loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_base_priors * num_classes, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        return reweight_factor * loss_cls,

    def loss_reg_single(self, anchors, bbox_pred, labels,
                        label_weights, bbox_targets, bbox_weights,
                        reweight_factor, num_total_samples):
        """Compute reg loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 5).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_base_priors * 4, H, W).
            iou_pred (Tensor): Iou for a single scale level, the
                channel number is (N, num_base_priors * 1, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox weights of all anchors in the
                image with shape (N, 4)
            reweight_factor (list[int]): Reweight factor for cls and reg
                loss.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        anchors = anchors.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)
        # iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1, )
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # iou_targets = label_weights.new_zeros(labels.shape)
        # iou_weights = label_weights.new_zeros(labels.shape)
        # iou_weights[(bbox_weights.sum(axis=1) > 0).nonzero(
        #     as_tuple=False)] = 1.

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    &
                    (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred.tensor,
                pos_decode_bbox_targets.tensor,
                # bbox_weights,
                avg_factor=num_total_samples)

            # iou_targets[pos_inds] = bbox_overlaps(
            #     pos_decode_bbox_pred.detach(),
            #     pos_decode_bbox_targets,
            #     is_aligned=True)
            # loss_iou = self.loss_iou(
            #     iou_pred,
            #     iou_targets,
            #     iou_weights,
            #     avg_factor=num_total_samples)
        else:
            loss_bbox = bbox_pred.sum() * 0
            # loss_iou = iou_pred.sum() * 0

        return reweight_factor * loss_bbox,

    def feature_refine(self, x: List[Tensor],
                       rois: List[List[Tensor]]) -> List[Tensor]:
        """Refine the input feature use feature refine module.

        Args:
            x (list[Tensor]): feature maps of multiple scales.
            rois (list[list[Tensor]]): input rbboxes of multiple
                scales of multiple images, output by former stages
                and are to be refined.

        Returns:
            list[Tensor]: refined feature maps of multiple scales.
        """
        return self.feat_refine_module(x, rois)

    def refine_bboxes(self, cls_scores: List[Tensor], bbox_preds: List[Tensor],
                      rois: List[List[Tensor]]) -> List[List[Tensor]]:
        """Refine predicted bounding boxes at each position of the feature
        maps. This method will be used in R3Det in refinement stages.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)
            rois (list[list[Tensor]]): input rbboxes of each level of each
                image. rois output by former stages and are to be refined

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level of each
            image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]

        assert rois is not None
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]
            assert bbox_pred.size(1) == 5
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, 5)
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 5)
            for img_id in range(num_imgs):
                bboxes_list[img_id].append(refined_bbox[img_id].detach())
        return bboxes_list

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            cls_scores,
                            bbox_preds,
                            num_level_anchors,
                            # gt_bboxes,
                            gt_bboxes_ignore,
                            # gt_labels,
                            img_meta,
                            batch_gt_instances,
                            label_channels=1,
                            unmap_outputs=True,
                            is_cls_assigner=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        bbox_preds_valid = bbox_preds[inside_flags, :]
        cls_scores_valid = cls_scores[inside_flags, :]
        assigner = self.cls_assigner if is_cls_assigner else self.reg_assigner
        bbox_preds_valid = self.bbox_coder.decode(anchors, bbox_preds_valid)

        pred_instances = InstanceData(priors=anchors, bboxes=bbox_preds_valid, scores=cls_scores_valid)
        gt_instances = batch_gt_instances
        # pred_instances['priors'] = anchors
        # pred_instances['bboxes'] = bbox_preds_valid
        # pred_instances['scores'] = cls_scores_valid
        # gt_instances['bboxes'] = gt_bboxes
        # gt_instances['labels'] = gt_labels
        assign_result = assigner.assign(pred_instances, num_level_anchors_inside,
                                        gt_instances, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        # pred_instances = InstanceData(priors=anchors)
        # assign_result = self.assigner.assign(pred_instances, gt_instances,
        #                                      gt_bboxes_ignore)

        # sampling_result = self.sampler.sample(assign_result, pred_instances,
        #                                       gt_instances)

        num_valid_anchors = anchors.shape[0]
        # target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox \
        #     else self.bbox_coder.encode_size
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
                    RotatedBoxes(sampling_result.pos_bboxes), sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            gt_labels = gt_instances.labels
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    # def get_targets(self,
    #                 anchor_list,
    #                 valid_flag_list,
    #                 batch_gt_instances,
    #                 img_metas,
    #                 gt_bboxes_ignore_list=None,
    #                 # gt_labels_list=None,
    #                 label_channels=1,
    #                 unmap_outputs=True,
    #                 return_sampling_results=False):
    #     """Compute regression and classification targets for anchors in
    #     multiple images.
    #
    #     Args:
    #         anchor_list (list[list[Tensor]]): Multi level anchors of each
    #             image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, 4).
    #         valid_flag_list (list[list[Tensor]]): Multi level valid flags of
    #             each image. The outer list indicates images, and the inner list
    #             corresponds to feature levels of the image. Each element of
    #             the inner list is a tensor of shape (num_anchors, )
    #         gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
    #         img_metas (list[dict]): Meta info of each image.
    #         gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
    #             ignored.
    #         gt_labels_list (list[Tensor]): Ground truth labels of each box.
    #         label_channels (int): Channel of label.
    #         unmap_outputs (bool): Whether to map outputs back to the original
    #             set of anchors.
    #
    #     Returns:
    #         tuple: Usually returns a tuple containing learning targets.
    #
    #             - labels_list (list[Tensor]): Labels of each level.
    #             - label_weights_list (list[Tensor]): Label weights of each
    #               level.
    #             - bbox_targets_list (list[Tensor]): BBox targets of each level.
    #             - bbox_weights_list (list[Tensor]): BBox weights of each level.
    #             - num_total_pos (int): Number of positive samples in all
    #               images.
    #             - num_total_neg (int): Number of negative samples in all
    #               images.
    #
    #         additional_returns: This function enables user-defined returns from
    #             `self._get_targets_single`. These returns are currently refined
    #             to properties at each feature map (i.e. having HxW dimension).
    #             The results will be concatenated after the end
    #     """
    #     num_imgs = len(img_metas)
    #     assert len(anchor_list) == len(valid_flag_list) == num_imgs
    #     if gt_bboxes_ignore_list is None:
    #         batch_gt_instances_ignore = [None] * num_imgs
    #
    #     # anchor number of multi levels
    #     num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    #     # concat all level anchors to a single tensor
    #     concat_anchor_list = []
    #     concat_valid_flag_list = []
    #     for i in range(num_imgs):
    #         assert len(anchor_list[i]) == len(valid_flag_list[i])
    #         concat_anchor_list.append(cat_boxes(anchor_list[i]))
    #         concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
    #
    #     # compute targets for each image
    #     # if gt_bboxes_ignore_list is None:
    #     #     gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    #     # if gt_labels_list is None:
    #     #     gt_labels_list = [None for _ in range(num_imgs)]
    #     results = multi_apply(
    #         self._get_targets_single,
    #         concat_anchor_list,
    #         concat_valid_flag_list,
    #         batch_gt_instances,
    #         img_metas,
    #         # gt_bboxes_list,
    #         gt_bboxes_ignore_list,
    #         # gt_labels_list,
    #         label_channels=label_channels,
    #         unmap_outputs=unmap_outputs)
    #     (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
    #      pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
    #     rest_results = list(results[7:])  # user-added return values
    #     # avg_factor = sum(
    #     #     [results.avg_factor for results in sampling_results_list])
    #     self._raw_positive_infos.update(sampling_results=sampling_results_list)
    #
    #     # no valid anchors
    #     if any([labels is None for labels in all_labels]):
    #         return None
    #     # sampled anchors of all images
    #     num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    #     num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    #     # split targets to a list w.r.t. multiple levels
    #     labels_list = images_to_levels(all_labels, num_level_anchors)
    #     label_weights_list = images_to_levels(all_label_weights,
    #                                           num_level_anchors)
    #     bbox_targets_list = images_to_levels(all_bbox_targets,
    #                                          num_level_anchors)
    #     bbox_weights_list = images_to_levels(all_bbox_weights,
    #                                          num_level_anchors)
    #     res = (labels_list, label_weights_list, bbox_targets_list,
    #            bbox_weights_list, num_total_pos, num_total_neg)
    #     if return_sampling_results:
    #         res = res + (sampling_results_list,)
    #     for i, r in enumerate(rest_results):  # user-added return values
    #         rest_results[i] = images_to_levels(r, num_level_anchors)
    #
    #     return res + tuple(rest_results)

    def get_cls_targets(self,
                        anchor_list,
                        valid_flag_list,
                        num_level_anchors_list,
                        cls_score_list,
                        bbox_pred_list,
                        gt_bboxes_list,
                        img_metas,
                        batch_gt_instances,
                        gt_bboxes_ignore_list=None,
                        gt_labels_list=None,
                        label_channels=1,
                        unmap_outputs=True):
        """Get cls targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()`.
        Besides returning the targets as the parent  method does,
        it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors_list (list[Tensor]): Number of anchors of each
                scale level of all image.
            cls_score_list (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_pred_list (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Return:
            tuple[Tensor]: A tuple of cls targets components.
        """
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
            self._get_targets_single,
            anchor_list,
            valid_flag_list,
            cls_score_list,
            bbox_pred_list,
            num_level_anchors_list,
            # gt_bboxes_list,
            gt_bboxes_ignore_list,
            # gt_labels_list,
            img_metas,
            batch_gt_instances,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
            is_cls_assigner=True)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        # `avg_factor` is usually equal to the number of positive priors.
        # avg_factor = sum(
        #     [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def get_reg_targets(self,
                        anchor_list,
                        valid_flag_list,
                        num_level_anchors_list,
                        cls_score_list,
                        bbox_pred_list,
                        # gt_bboxes_list,
                        img_metas,
                        batch_gt_instances,
                        gt_bboxes_ignore_list=None,
                        # gt_labels_list=None,
                        label_channels=1,
                        unmap_outputs=True):
        """Get reg targets for DDOD head.

        This method is almost the same as `AnchorHead.get_targets()` when
        is_cls_assigner is False. Besides returning the targets as the parent
        method does, it also returns the anchors as the first element of the
        returned tuple.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            num_level_anchors (int): Number of anchors of each scale level.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            gt_labels_list (list[Tensor]): class indices corresponding to
                each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of reg targets components.
        """
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list,
         sampling_results_list) = multi_apply(
            self._get_targets_single,
            anchor_list,
            valid_flag_list,
            cls_score_list,
            bbox_pred_list,
            num_level_anchors_list,
            # gt_bboxes_list,
            gt_bboxes_ignore_list,
            # gt_labels_list,
            img_metas,
            batch_gt_instances,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
            is_cls_assigner=False)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # `avg_factor` is usually equal to the number of positive priors.
        # avg_factor = sum(
        #     [results.avg_factor for results in sampling_results_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors_list[0])
        labels_list = images_to_levels(all_labels, num_level_anchors_list[0])
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors_list[0])
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors_list[0])
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors_list[0])
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def process_predictions_and_anchors(self, anchor_list, valid_flag_list,
                                        cls_scores, bbox_preds, img_metas,
                                        gt_bboxes_ignore_list):
        """Compute common vars for regression and classification targets.

        Args:
            anchor_list (list[Tensor]): anchors of each image.
            valid_flag_list (list[Tensor]): Valid flags of each image.
            cls_scores (list[Tensor]): Classification scores for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for all scale
                levels, each is a 4D-tensor, the channels number is
                num_base_priors * 4.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore_list (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Return:
            tuple[Tensor]: A tuple of common loss vars.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs
        # anchor_list_my = []
        anchor_list_ = []

        valid_flag_list_ = []
        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])

            b0 = anchor_list[i][0].tensor
            b1 = anchor_list[i][1].tensor
            b2 = anchor_list[i][2].tensor
            b3 = anchor_list[i][3].tensor
            b4 = anchor_list[i][4].tensor
            anchor_list_.append(torch.cat([b0, b1, b2, b3, b4], dim=0))
            # lllll?????????
            # anchor_list_.append(torch.cat(anchor_list_my[i]))
            valid_flag_list_.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]

        num_levels = len(cls_scores)
        cls_score_list = []
        bbox_pred_list = []

        mlvl_cls_score_list = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.num_base_priors * self.cls_out_channels)
            for cls_score in cls_scores
        ]
        mlvl_bbox_pred_list = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_base_priors * 5)
            for bbox_pred in bbox_preds
        ]

        for i in range(num_imgs):
            mlvl_cls_tensor_list = [
                mlvl_cls_score_list[j][i] for j in range(num_levels)
            ]
            mlvl_bbox_tensor_list = [
                mlvl_bbox_pred_list[j][i] for j in range(num_levels)
            ]
            cat_mlvl_cls_score = torch.cat(mlvl_cls_tensor_list, dim=0)
            cat_mlvl_bbox_pred = torch.cat(mlvl_bbox_tensor_list, dim=0)
            cls_score_list.append(cat_mlvl_cls_score)
            bbox_pred_list.append(cat_mlvl_bbox_pred)
        return (anchor_list_, valid_flag_list_, num_level_anchors_list,
                cls_score_list, bbox_pred_list, gt_bboxes_ignore_list)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        """Get the anchors of each scale level inside.

        Args:
            num_level_anchors (list[int]): Number of anchors of each
                scale level.
            inside_flags (Tensor): Multi level inside flags of the image,
                which are concatenated into a single tensor of
                shape (num_base_priors,).

        Returns:
            list[int]: Number of anchors of each scale level inside.
        """
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
