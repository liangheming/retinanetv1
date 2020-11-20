from typing import List, Any

import torch
from utils.boxs_utils import box_iou
from losses.commons import IOULoss


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_reg[..., :2] = anchors_xy + scale_reg[..., :2] * anchors_wh
        scale_reg[..., 2:] = scale_reg[..., 2:].exp() * anchors_wh
        scale_reg[..., :2] -= (0.5 * scale_reg[..., 2:])
        scale_reg[..., 2:] = scale_reg[..., :2] + scale_reg[..., 2:]

        return scale_reg


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, iou_thresh=0.5, ignore_iou=0.4, allow_low_quality_matches=True):
        self.iou_thresh = iou_thresh
        self.ignore_iou = ignore_iou
        self.allow_low_quality_matches = allow_low_quality_matches

    @torch.no_grad()
    def __call__(self, anchors, gt_boxes):
        ret = list()
        for idx, gt_box in enumerate(gt_boxes):
            if len(gt_box) == 0:
                continue
            ori_match = None
            gt_anchor_iou = box_iou(gt_box[..., 1:], anchors)
            match_val, match_idx = gt_anchor_iou.max(dim=0)
            if self.allow_low_quality_matches:
                ori_match = match_idx.clone()
            match_idx[match_val < self.ignore_iou] = self.BELOW_LOW_THRESHOLD
            match_idx[(match_val >= self.ignore_iou) & (match_val < self.iou_thresh)] = self.BETWEEN_THRESHOLDS
            if self.allow_low_quality_matches:
                self.set_low_quality_matches_(match_idx, ori_match, gt_anchor_iou)
            ret.append((idx, match_idx))
        return ret

    @staticmethod
    def set_low_quality_matches_(matches, ori_matches, gt_anchor_iou):
        highest_quality_foreach_gt, _ = gt_anchor_iou.max(dim=1)
        # [num,2](gt_idx,anchor_idx)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            gt_anchor_iou == highest_quality_foreach_gt[:, None], as_tuple=False
        )
        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = ori_matches[pred_inds_to_update]


def focal_loss(predicts, targets, alpha=0.25, gamma=2.0):
    pos_loss = -alpha * targets * (
            (1 - predicts) ** gamma) * predicts.log()
    neg_loss = - (1 - alpha) * (1. - targets) * (predicts ** gamma) * (
        (1 - predicts).log())
    return pos_loss + neg_loss


class RetinaLoss(object):
    def __init__(self, iou_thresh=0.5,
                 ignore_thresh=0.4,
                 alpha=0.25,
                 gamma=2.0,
                 iou_type="giou",
                 allow_low_quality_matches=False):
        self.iou_thresh = iou_thresh
        self.ignore_thresh = ignore_thresh
        self.alpha = alpha
        self.gamma = gamma
        self.mather = Matcher(iou_thresh, ignore_thresh, allow_low_quality_matches)
        self.box_coder = BoxCoder()
        self.iou_loss = IOULoss(iou_type=iou_type, coord_type="xyxy")

    def __call__(self, cls_predicts, reg_predicts, anchors, targets):
        """
        """
        cls_predicts = torch.cat([item for item in cls_predicts], dim=1)
        reg_predicts = torch.cat([item for item in reg_predicts], dim=1)
        all_anchors = torch.cat([item for item in anchors])
        gt_boxes = targets['target'].split(targets['batch_len'])
        match_ret = self.mather(all_anchors, gt_boxes)
        all_batch_idx = list()
        all_anchor_idx = list()
        all_cls_target = list()
        all_box_target = list()
        for bid, match in match_ret:
            positive_anchor_idx = (match >= 0).nonzero(as_tuple=False).squeeze(-1)
            negative_anchor_idx = (match == self.mather.BELOW_LOW_THRESHOLD).nonzero(as_tuple=False).squeeze(-1)
            gt_idx = match[positive_anchor_idx]
            target_label = gt_boxes[bid][gt_idx, 0].long()
            target_box = gt_boxes[bid][gt_idx, 1:]
            cls_target = torch.zeros(size=(len(positive_anchor_idx) + len(negative_anchor_idx),
                                           cls_predicts.shape[-1]),
                                     device=cls_predicts.device)
            cls_target[range(len(positive_anchor_idx)), target_label] = 1.
            all_cls_target.append(cls_target)
            all_box_target.append(target_box)
            all_batch_idx.append([bid] * len(positive_anchor_idx))
            all_batch_idx.append([bid] * len(negative_anchor_idx))
            all_anchor_idx.append(positive_anchor_idx)
            all_anchor_idx.append(negative_anchor_idx)
        all_cls_target = torch.cat(all_cls_target, dim=0)
        all_box_target = torch.cat(all_box_target, dim=0)
        all_cls_predict = cls_predicts[sum(all_batch_idx, []),
                                       torch.cat(all_anchor_idx)]
        if all_cls_predict.dtype == torch.float16:
            all_cls_predict = all_cls_predict.float()
        all_cls_predict = all_cls_predict.sigmoid()
        cls_loss = focal_loss(all_cls_predict, all_cls_target, self.alpha, self.gamma).sum()
        all_reg_predicts = reg_predicts[sum(all_batch_idx[::2], []), torch.cat(all_anchor_idx[::2])]
        predict_box = self.box_coder.decoder(all_reg_predicts, all_anchors[torch.cat(all_anchor_idx[::2])])
        iou_loss = self.iou_loss(predict_box, all_box_target).sum()

        num_of_positive = len(predict_box)
        return cls_loss / num_of_positive, iou_loss / num_of_positive, num_of_positive
