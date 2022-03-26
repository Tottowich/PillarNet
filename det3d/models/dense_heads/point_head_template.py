import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.bbox import box_torch_ops
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_single_targets(self, points, gt_boxes, extend_gt_boxes=None,
                              set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N, 3) [x, y, z]
            gt_boxes: (M, 8)
            extend_gt_boxes: (M, 8)
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N), long type, 0:background, -1:ignored
            point_box_labels: (N, code_size)
        """
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        # point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None

        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            points.view(len(gt_boxes), -1, points.shape[-1]), gt_boxes[:, 0:7].contiguous()
        ).long().squeeze(dim=0)
        box_fg_flag = (box_idxs_of_pts >= 0)
        if set_ignore_flag:
            extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points.unsqueeze(dim=0), extend_gt_boxes[:, 0:7].contiguous()
            ).long().squeeze(dim=0)
            fg_flag = box_fg_flag
            ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
            point_cls_labels[ignore_flag] = -1
        elif use_ball_constraint:
            box_centers = gt_boxes[box_idxs_of_pts][:, 0:3].clone()
            box_centers[:, 2] += gt_boxes[box_idxs_of_pts][:, 5] / 2
            ball_flag = ((box_centers - points).norm(dim=1) < central_radius)
            fg_flag = box_fg_flag & ball_flag
        else:
            raise NotImplementedError

        gt_box_of_fg_points = gt_boxes[box_idxs_of_pts[fg_flag]]
        point_cls_labels[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()

        # if ret_part_labels:
        #     transformed_points = points[fg_flag] - gt_box_of_fg_points[:, 0:3]
        #     transformed_points = box_torch_ops.rotate_points_along_z(
        #         transformed_points.view(-1, 1, 3), gt_box_of_fg_points[:, 6]
        #     ).view(-1, 3)
        #     offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
        #     point_part_labels[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            # 'point_part_labels': point_part_labels
        }
        return targets_dict
