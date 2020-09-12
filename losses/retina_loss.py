import torch
from commons.boxs_utils import box_iou
from utils.retinanet import BoxCoder
from losses.commons import IOULoss


def get_gpu_num():
    import os
    g_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if 'CUDA_VISIBLE_DEVICES' in os.environ else 1
    return g_num


def reduce_sum(tensor):
    import torch.distributed as dist
    g_num = get_gpu_num()
    if g_num <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


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
    def __init__(self, iou_thresh=0.5, ignore_thresh=0.4, alpha=0.25, gamma=2.0, iou_type="giou"):
        self.iou_thresh = iou_thresh
        self.ignore_thresh = ignore_thresh
        self.alpha = alpha
        self.gama = gamma
        self.builder = RetinaLossBuilder(iou_thresh, ignore_thresh)
        self.box_coder = BoxCoder()
        self.iou_loss = IOULoss(iou_type=iou_type, coord_type="xyxy")

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
        cls_loss_list = list()
        reg_loss_list = list()

        pos_num_sum = 0.
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

            mix_weights = torch.ones(size=valid_cls_predicts.shape, device=device)
            mix_weights[range(pos_num), gt[pos_idx, 1].long()] = gt[pos_idx, 0]
            pos_loss = -self.alpha * cls_targets * ((1 - valid_cls_predicts) ** self.gama) * valid_cls_predicts.log()
            pos_loss = mix_weights * pos_loss
            neg_loss = -(1 - self.alpha) * (1. - cls_targets) * (valid_cls_predicts ** self.gama) * (
                (1 - valid_cls_predicts).log())
            cls_loss = (pos_loss + neg_loss).sum()
            cls_loss_list.append(cls_loss)

            valid_reg_predicts = batch_reg_predict[pos_idx, :]
            predict_box = self.box_coder.decoder(valid_reg_predicts, all_anchors[pos_idx])
            gt_bbox = gt[pos_idx, 2:]
            # delta_targets = self.box_coder.encoder(all_anchors[pos_idx], gt_bbox)
            reg_loss = self.iou_loss(predict_box, gt_bbox)
            reg_loss * gt[pos_idx, 0]
            reg_loss_list.append(reg_loss.sum())

        cls_loss_sum = torch.stack(cls_loss_list).sum()
        # pos_num_sum = reduce_sum(torch.tensor(data=pos_num_sum, device=device).float()).item() / get_gpu_num()
        if pos_num_sum == 0:
            total_loss = cls_loss_sum
            return total_loss, torch.stack([cls_loss_sum, torch.tensor(data=0., device=device)]).detach(), pos_num_sum
        reg_loss_sum = torch.stack(reg_loss_list).sum()

        cls_loss_mean = cls_loss_sum / pos_num_sum
        reg_loss_mean = reg_loss_sum / pos_num_sum
        total_loss = cls_loss_mean + reg_loss_mean

        return total_loss, torch.stack([cls_loss_mean, reg_loss_mean]).detach(), pos_num_sum
