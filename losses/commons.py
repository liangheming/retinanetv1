import torch
import math


#
# class BoxSimilarity(object):
#     def __init__(self, iou_type="giou"):
#         super(BoxSimilarity, self).__init__()
#         self.iou_type = iou_type
#
#     def __call__(self, predicts, targets):
#         """
#         :param predicts: [box_num,4] (x1,y1,x2,y2)
#         :param targets: [box_num,4] (x1,y1,x2,y2)
#         :return:
#         """
#         p_x1, p_y1, p_x2, p_y2 = predicts.t()
#         t_x1, t_y1, t_x2, t_y2 = targets.t()
#         i_x1 = torch.max(p_x1, t_x1)
#         i_y1 = torch.max(p_y1, t_y1)
#         i_x2 = torch.min(p_x2, t_x2)
#         i_y2 = torch.min(p_y2, t_y2)
#         inter_area = (i_x2 - i_x1).clamp(min=0) * (i_y2 - i_y1).clamp(min=0)
#         p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
#         t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
#         union_area = (p_area + t_area - inter_area + 1e-6)
#         ious = inter_area / union_area
#         if self.iou_type == 'iou':
#             return ious
#         c_x1 = torch.min(p_x1, t_x1)
#         c_x2 = torch.max(p_x2, t_x2)
#         c_y1 = torch.min(p_y1, t_y1)
#         c_y2 = torch.max(p_y2, t_y2)
#         cw = c_x2 - c_x1
#         ch = c_y2 - c_y1
#         c_area = cw * ch + 1e-6
#         if self.iou_type == "giou":
#             giou = ious - (c_area - union_area) / c_area
#             return giou
#         p_center_x = (p_x1 + p_x2) / 2.
#         p_center_y = (p_y1 + p_y2) / 2.
#         t_center_x = (t_x1 + t_x2) / 2.
#         t_center_y = (t_y1 + t_y2) / 2.
#
#         center_distance = (t_center_x - p_center_x) ** 2 + (t_center_y - p_center_y) ** 2
#         diagonal_distance = cw ** 2 + ch ** 2 + 1e-6
#         if self.iou_type == "diou":
#             diou = ious - center_distance / diagonal_distance
#             return diou
#         t_w = t_x2 - t_x1
#         t_h = t_y2 - t_y1
#         p_w = p_x2 - p_x1
#         p_h = p_y2 - p_y1
#
#         v = (4 / math.pi ** 2) * (torch.atan(t_w / t_h) - torch.atan(p_w / p_h)) ** 2
#         with torch.no_grad():
#             alpha = v / (1 - ious + v + 1e-6)
#         if self.iou_type == "ciou":
#             ciou = ious - (center_distance / diagonal_distance + v * alpha)
#             return ciou
#         raise NotImplementedError("{:s} is not support".format(self.iou_type))
#
#
# class IOULoss(object):
#     def __init__(self, iou_type="giou"):
#         super(IOULoss, self).__init__()
#         self.iou_type = iou_type
#         self.box_distance = BoxSimilarity(iou_type)
#
#     def __call__(self, predicts, targets):
#         dis = self.box_distance(predicts, targets)
#         if self.iou_type == 'iou':
#             return -dis.log()
#         else:
#             return 1 - dis
class BoxSimilarity(object):
    def __init__(self, iou_type="giou", coord_type="xyxy", eps=1e-9):
        self.iou_type = iou_type
        self.coord_type = coord_type
        self.eps = eps

    def __call__(self, box1, box2):
        """
        :param box1: [num,4] predicts
        :param box2:[num,4] targets
        :return:
        """
        box1_t = box1.T
        box2_t = box2.T

        if self.coord_type == "xyxy":
            b1_x1, b1_y1, b1_x2, b1_y2 = box1_t[0], box1_t[1], box1_t[2], box1_t[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2_t[0], box2_t[1], box2_t[2], box2_t[3]
        elif self.coord_type == "xywh":
            b1_x1, b1_x2 = box1_t[0] - box1_t[2] / 2., box1_t[0] + box1_t[2] / 2.
            b1_y1, b1_y2 = box1_t[1] - box1_t[3] / 2., box1_t[1] + box1_t[3] / 2.
            b2_x1, b2_x2 = box2_t[0] - box2_t[2] / 2., box2_t[0] + box2_t[2] / 2.
            b2_y1, b2_y2 = box2_t[1] - box2_t[3] / 2., box2_t[1] + box2_t[3] / 2.
        elif self.coord_type == "ltrb":
            b1_x1, b1_y1 = 0. - box1_t[0], 0. - box1_t[1]
            b1_x2, b1_y2 = 0. + box1_t[2], 0. + box1_t[3]
            b2_x1, b2_y1 = 0. - box2_t[0], 0. - box2_t[1]
            b2_x2, b2_y2 = 0. + box2_t[2], 0. + box2_t[3]
        else:
            raise NotImplementedError("coord_type only support xyxy, xywh,ltrb")
        inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                     (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union_area = w1 * h1 + w2 * h2 - inter_area + self.eps
        iou = inter_area / union_area
        if self.iou_type == "iou":
            return iou

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if self.iou_type == "giou":
            c_area = cw * ch + self.eps
            giou = iou - (c_area - union_area) / c_area
            return giou

        diagonal_dis = cw ** 2 + ch ** 2 + self.eps
        center_dis = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                      (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        if self.iou_type == 'diou':
            diou = iou - center_dis / diagonal_dis
            return diou

        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        with torch.no_grad():
            alpha = v / ((1 + self.eps) - iou + v)

        if self.iou_type == "ciou":
            ciou = iou - (center_dis / diagonal_dis + v * alpha)
            return ciou

        raise NotImplementedError("iou_type only support iou,giou,diou,ciou")


class IOULoss(object):
    def __init__(self, iou_type="giou", coord_type="xyxy"):
        super(IOULoss, self).__init__()
        self.iou_type = iou_type
        self.box_similarity = BoxSimilarity(iou_type, coord_type)

    def __call__(self, predicts, targets):
        similarity = self.box_similarity(predicts, targets)
        if self.iou_type == "iou":
            return -similarity.log()
        else:
            return 1 - similarity
