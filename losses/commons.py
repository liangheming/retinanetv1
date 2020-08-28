import torch
import math


class IOULoss(object):
    def __init__(self, iou_type="giou"):
        super(IOULoss, self).__init__()
        self.iou_type = iou_type

    def __call__(self, predicts, targets):
        """
        :param predicts: [box_num,4] (x1,y1,x2,y2)
        :param targets: [box_num,4] (x1,y1,x2,y2)
        :return:
        """
        p_x1, p_y1, p_x2, p_y2 = predicts.t()
        t_x1, t_y1, t_x2, t_y2 = targets.t()
        i_x1 = torch.max(p_x1, t_x1)
        i_y1 = torch.max(p_y1, t_y1)
        i_x2 = torch.min(p_x2, t_x2)
        i_y2 = torch.min(p_y2, t_y2)
        inter_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
        p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
        union_area = (p_area + t_area - inter_area + 1e-6)
        ious = inter_area / union_area
        if self.iou_type == 'iou':
            return -ious.log()
        c_x1 = torch.min(p_x1, t_x1)
        c_x2 = torch.max(p_x2, t_x2)
        c_y1 = torch.min(p_y1, t_y1)
        c_y2 = torch.max(p_y2, t_y2)
        cw = c_x2 - c_x1
        ch = c_y2 - c_y1
        c_area = cw * ch + 1e-6
        if self.iou_type == "giou":
            giou = ious - (c_area - union_area) / c_area
            return 1 - giou
        p_center_x = (p_x1 + p_x2) / 2.
        p_center_y = (p_y1 + p_y2) / 2.
        t_center_x = (t_x1 + t_x2) / 2.
        t_center_y = (t_y1 + t_y2) / 2.

        center_distance = (t_center_x - p_center_x) ** 2 + (t_center_y - p_center_y) ** 2
        diagonal_distance = cw ** 2 + ch ** 2 + 1e-6
        if self.iou_type == "diou":
            diou = ious - center_distance / diagonal_distance
            return 1 - diou
        t_w = t_x2 - t_x1
        t_h = t_y2 - t_y1
        p_w = p_x2 - p_x1
        p_h = p_y2 - p_y1

        v = (4 / math.pi ** 2) * (torch.atan(t_w / t_h) - torch.atan(p_w / p_h)) ** 2
        with torch.no_grad():
            alpha = v / (1 - ious + v + 1e-6)
        if self.iou_type == "ciou":
            ciou = ious - (center_distance / diagonal_distance + v * alpha)
            return 1 - ciou
        raise NotImplementedError("{:s} is not support".format(self.iou_type))
