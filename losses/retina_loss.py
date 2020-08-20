import torch
from utils.boxs import box_iou


def smooth_l1_loss(predicts, target, beta=1. / 9):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(predicts - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss


class RetinaLossBuilder(object):
    def __init__(self, iou_thresh=0.5, ignore_iou=0.4):
        self.iou_thresh = iou_thresh
        self.ignore_iou = ignore_iou

    @torch.no_grad()
    def __call__(self, bs, anchors, targets):
        """
        :param bs: batch_size
        :param anchors: list(anchor) anchor [all, 4] (x1,y1,x2,y2)
        :param targets: [gt_num, 7] (batch_id,weights,label_id,x1,y1,x2,y2)
        :return:
        """
        # [all,4] (x1,y1,x2,y2)
        all_anchors = torch.cat(anchors, dim=0)
        flag_list = list()
        targets_list = list()
        for bi in range(bs):
            flag = torch.ones(size=(len(all_anchors),), device=all_anchors.device)
            # flag = all_anchors.new_ones(size=(len(all_anchors),))
            # [gt_num, 6] (weights,label_idx,x1,y1,x2,y2)
            batch_targets = targets[targets[:, 0] == bi, 1:]
            if len(batch_targets) == 0:
                flag_list.append(flag * 0.)
                targets_list.append(torch.Tensor())
                continue
            flag *= -1.
            batch_box = batch_targets[:, 2:]
            # [all,gt_num]
            anchor_gt_iou = box_iou(all_anchors, batch_box)

            iou_val, gt_idx = anchor_gt_iou.max(dim=1)
            pos_idx = iou_val >= self.iou_thresh
            neg_idx = iou_val < self.ignore_iou
            flag[pos_idx] = 1.
            flag[neg_idx] = 0.
            flag_list.append(flag)
            gt_targets = batch_targets[gt_idx, :]
            targets_list.append(gt_targets)
        return flag_list, targets_list, all_anchors


class RetinaLoss(object):
    def __init__(self, iou_thresh=0.5, ignore_thresh=0.4, alpha=0.25, gamma=2.0, beta=1. / 9):
        self.iou_thresh = iou_thresh
        self.ignore_thresh = ignore_thresh
        self.alpha = alpha
        self.gama = gamma
        self.beta = beta
        self.builder = RetinaLossBuilder(iou_thresh, ignore_thresh)
        self.std = torch.tensor([0.1, 0.1, 0.2, 0.2]).float()

    def __call__(self, cls_predicts, reg_predicts, anchors, targets):
        """
        :param cls_predicts: list(cls_predict) cls_predict[bs,all,num_cls]
        :param reg_predicts: list(reg_predict) reg_predict[bs,all,4]
        :param anchors: list(anchor) anchor[all,4]
        :param targets: [gt_num,7] (batch_id,weights,label_id,x1,y1,x2,y2)
        :return:
        """
        for i in range(len(cls_predicts)):
            if cls_predicts[i].dtype == torch.float16:
                cls_predicts[i] = cls_predicts[i].float()
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        flags, gt_targets, all_anchors = self.builder(bs, anchors, targets)
        anchors_wh = all_anchors[:, [2, 3]] - all_anchors[:, [0, 1]]
        anchors_xy = all_anchors[:, [0, 1]] + 0.5 * anchors_wh
        std = self.std.to(device)
        cls_loss_list = list()
        reg_loss_list = list()

        pos_num_sum = 0
        for bi in range(bs):
            batch_cls_predict = torch.cat([cls_item[bi] for cls_item in cls_predicts], dim=0) \
                .sigmoid() \
                .clamp(1e-6, 1 - 1e-6)
            batch_reg_predict = torch.cat([reg_item[bi] for reg_item in reg_predicts], dim=0)
            flag = flags[bi]
            gt = gt_targets[bi]
            pos_idx = (flag == 1).nonzero(as_tuple=False).squeeze(1)
            pos_num = len(pos_idx)
            if pos_num == 0:
                neg_cls_loss = -(1 - self.alpha) * batch_cls_predict ** self.gama * ((1 - batch_cls_predict).log())
                cls_loss_list.append(neg_cls_loss.sum())
                continue
            pos_num_sum += pos_num
            neg_idx = (flag == 0).nonzero(as_tuple=False).squeeze(1)
            valid_idx = torch.cat([pos_idx, neg_idx])
            valid_cls_predicts = batch_cls_predict[valid_idx, :]
            cls_targets = torch.zeros(size=valid_cls_predicts.shape, device=device)
            cls_targets[range(pos_num), gt[pos_idx, 1].long()] = 1.
            pos_loss = -self.alpha * cls_targets * ((1 - valid_cls_predicts) ** self.gama) * valid_cls_predicts.log()
            neg_loss = -(1 - self.alpha) * (1. - cls_targets) * (valid_cls_predicts ** self.gama) * (
                (1 - valid_cls_predicts).log())
            cls_loss = (pos_loss + neg_loss).sum()
            cls_loss_list.append(cls_loss)

            valid_reg_predicts = batch_reg_predict[pos_idx, :]
            gt_bbox = gt[pos_idx, 2:]
            valid_anchor_wh = anchors_wh[pos_idx, :]
            valid_anchor_xy = anchors_xy[pos_idx, :]

            gt_wh = (gt_bbox[:, [2, 3]] - gt_bbox[:, [0, 1]]).clamp(min=1.0)
            gt_xy = gt_bbox[:, [0, 1]] + 0.5 * gt_wh

            delta_xy = (gt_xy - valid_anchor_xy) / valid_anchor_wh
            delta_wh = (gt_wh / valid_anchor_wh).log()

            delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / std
            reg_loss = smooth_l1_loss(valid_reg_predicts, delta_targets, beta=self.beta).sum()
            reg_loss_list.append(reg_loss)

        cls_loss_sum = torch.stack(cls_loss_list).sum()
        if pos_num_sum == 0:
            total_loss = cls_loss_sum / bs
            return total_loss, torch.stack([cls_loss_sum, torch.tensor(data=0., device=device)]).detach(), pos_num_sum
        reg_loss_sum = torch.stack(reg_loss_list).sum()

        cls_loss_mean = cls_loss_sum / pos_num_sum
        reg_loss_mean = reg_loss_sum / pos_num_sum
        total_loss = cls_loss_mean + reg_loss_mean

        return total_loss, torch.stack([cls_loss_mean, reg_loss_mean]).detach(), pos_num_sum
