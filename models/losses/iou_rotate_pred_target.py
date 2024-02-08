import torch
import numpy as np
from .my_util import oriented_box_intersection_2d

def box2corners_th(box:torch.Te1nsor)-> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box (torch.Tensor): (B, N, 5) with x, y, w, h, alpha

    Returns:
        torch.Tensor: (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated




def cal_iou(box1: torch.Tensor, box2: torch.Tensor):
    """calculate iou

    Args:
        box1 (torch.Tensor): (B, N, 5)
        box2 (torch.Tensor): (B, N, 5)

    Returns:
        iou (torch.Tensor): (B, N)
        corners1 (torch.Tensor): (B, N, 4, 2)
        corners1 (torch.Tensor): (B, N, 4, 2)
        U (torch.Tensor): (B, N) area1 + area2 - inter_area
    """
    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)  # (B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = in1ter_area / u
    return iou


# import torch
#
# def compute_iou(pred, target):
#
#     pred_w = pred[:, 2] * torch.cos(pred[:, 4])
#     pred_h = pred[:, 3] * torch.sin(pred[:, 4])
#     pred_x1 = pred[:, 0] - pred_w / 2
#     pred_y1 = pred[:, 1] - pred_h / 2
#     pred_x2 = pred[:, 0] + pred_w / 2
#     pred_y2 = pred[:, 1] + pred_h / 2
#
#     #
#     target_w = target[:, 2] * torch.cos(target[:, 4])
#     target_h = target[:, 3] * torch.sin(target[:, 4])
#     target_x1 = target[:, 0] - target_w / 2
#     target_y1 = target[:, 1] - target_h / 2
#     target_x2 = target[:, 0] + target_w / 2
#     target_y2 = target[:, 1] + target_h / 2
#
#     #
#     inter_x1 = torch.max(pred_x1[:, None], target_x1)
#     inter_y1 = torch.max(pred_y1[:, None], target_y1)
#     inter_x2 = torch.min(pred_x2[:, None], target_x2)
#     inter_y2 = torch.min(pred_y2[:, None], target_y2)
#     inter_w = (inter_x2 - inter_x1 + 1).clamp(min=0)
#     inter_h = (inter_y2 - inter_y1 + 1).clamp(min=0)
#     inter_area = inter_w * inter_h
#
#     #
#     pred_w = pred[:, 2]
#     pred_h = pred[:, 3]
#     pred_area = pred_w * pred_h
#     target_w = target[:, 2]
#     target_h = target[:, 3]
#     target_area = target_w * target_h
#     union_area = pred_area[:, None] + target_area - inter_area
#
#     #
#     iou = inter_area / torch.clamp(union_area, min=1e-6)
#
#     return iou
# import torch
#
# EPSILON = 1e-8
#
#
# def iou_rotate_pred_target(pred, target):
#     """
#
#
#     :param pred: tensor, shape 为 [N, 5]
#     :param target: tensor, shape 为 [N, 5]
#     :return: IoU tensor, shape 为 [N, 1]
#     """
#
#     N = pred.shape[0]
#
#     # 将 [x, y, w, h, theta] 转换为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
#     # 这里通过下标来表示四个点，返回的是 shape 为 [N, 4, 2] 的 tensor。
#     pred_vertices = get_vertices_rotate_batch(pred[:, :2], pred[:, 2:4], pred[:, 4])
#     target_vertices = get_vertices_rotate_batch(target[:, :2], target[:, 2:4], target[:, 4])
#
#
#     vertices_intersect, area_intersect = overlaps(pred_vertices, target_vertices)
#
#     #
#     area_pred = pred[:, 2] * pred[:, 3]
#     area_target = target[:, 2] * target[:, 3]
#
#     #
#     area_union = area_pred + area_target - area_intersect
#     iou = torch.where(area_union <= 0, torch.tensor(0., dtype=torch.float32), (area_intersect + EPSILON) / (area_union + EPSILON))
#
#     return iou
#
#
#
#
#
# def get_vertices_rotate_batch(center, wh, angle):
#
#     w, h = wh[:, 0], wh[:, 1]
#     sin_t, cos_t = torch.sin(angle), torch.cos(angle)
#     rotation_matrix = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=1)
#     vertices_centered = torch.reshape(torch.stack([-w / 2, -h / 2, w / 2, h / 2], dim=1), shape=[-1, 2])  # 相对于中心点归一化后的矩形
#
#     #
#     vertices_rotated = torch.matmul(vertices_centered, rotation_matrix)
#     vertices = vertices_rotated + center[:, None, :] # 将中心点加回去
#
#     #
#     return torch.stack([vertices[:, :2], vertices[:, [0, 3]], vertices[:, 2:], vertices[:, [2, 1]]], dim=1)
#
#
#
# def overlaps(pred_vertices, target_vertices):
#     """
#
#     """
#     def polygon_area(vertices):
#         """
#
#         """
#         x, y = vertices[:, 0], vertices[:, 1]
#         return 0.5 * torch.abs(torch.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + x[-1] * y[0] - y[-1] * x[0])
#
#     batch_size = pred_vertices.shape[0]
#
#
#     def sort_vertices(vertices):
#         center = vertices.mean(dim=1)
#         angles = torch.atan2(vertices[:, :, 1] - center[:, None, 1], vertices[:, :, 0] - center[:, None, 0])
#         idx = torch.argsort(angles, dim=1)  # 按逆时针方向排序
#         return torch.gather(vertices, dim=1, index=idx.unsqueeze(-1).repeat(1, 1, 2))
#
#     pred_vertices_sorted = sort_vertices(pred_vertices)
#     target_vertices_sorted = sort_vertices(target_vertices)
#
#     vertices_intersect = []
#     for i in range(batch_size):
#
#         for j in range(4):
#             v1, v2, v3 = pred_vertices_sorted[i, j], pred_vertices_sorted[i, (j+1)%4], target_vertices_sorted[i, 0]
#             area_triange = polygon_area(torch.stack([v1, v2, v3], dim=0))
#             if area_triange > 0:
#                 continue
#
#             #
#             v1, v2, v3 = pred_vertices_sorted[i, j], pred_vertices_sorted[i, (j+1)%4], target_vertices_sorted[i, 1]
#             area_triange = polygon_area(torch.stack([v1, v2, v3], dim=0))
#             if area_triange > 0:
#                 vertices_intersect.append(target_vertices_sorted[i, 1])
#
#             #
#             v1, v2, v3 = pred_vertices_sorted[i, j], pred_vertices_sorted[i, (j+1)%4], target_vertices_sorted[i, 2]
#             area_triange = polygon_area(torch.stack([v1, v2, v3], dim=0))
#             if area_triange > 0:
#                 vertices_intersect.append(target_vertices_sorted[i, 2])
#
#     #
#     if not vertices_intersect:
#         return torch.zeros(batch_size)
#
#     vertices_intersect = torch.stack(vertices_intersect, dim=0)
#     area_intersection = polygon_area(vertices_intersect)
#
#     return area_intersection
