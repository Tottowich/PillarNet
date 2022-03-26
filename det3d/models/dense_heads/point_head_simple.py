import torch

from .point_head_template import PointHeadTemplate
from ..registry import POINT_HEAD


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    :param boxes3d: (M, 7)
    :param extra_width: [extra_x, extra_y, extra_z]
    :return:
    """
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


@POINT_HEAD.register_module
class PointHeadSimple(PointHeadTemplate):
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.in_planes = input_channels
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, example, batch_centers):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: [(N1, 3), (N2, 3), ...] [bs_idx, x, y, z]
                gt_boxes (optional): [(M1, 8), (M2, 8), ...]
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
        """
        gt_boxes = example['gt_box'][0]
        batch_size = len(gt_boxes)

        targets_dicts = []
        for k in range(batch_size):
            cur_gt_boxes = gt_boxes[k]
            extend_gt_boxes = enlarge_box3d(
                cur_gt_boxes.view(-1, cur_gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
            ).view(-1, cur_gt_boxes.shape[-1])

            targets_dicts.append(self.assign_single_targets(
                points=batch_centers[k], gt_boxes=cur_gt_boxes, extend_gt_boxes=extend_gt_boxes,
                set_ignore_flag=True, use_ball_constraint=False
            ))

        return targets_dicts

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, example, batch_centers, training=True):
        point_features = example['features']

        batch_size = len(point_features)
        example['point_cls_scores'] = []
        for k in range(batch_size):
            point_cls_preds = self.cls_layers(point_features[k].view(-1, self.in_planes))  # (num_rois, num_points, num_class)
            point_cls_scores = torch.sigmoid(point_cls_preds)
            # point_cls_scores, _ = point_cls_scores.max(dim=-1)
            example['point_cls_scores'].append(point_cls_scores.view(*point_features[k].shape[:2], 1))

        if self.training:
            targets_dict = self.assign_targets(example, batch_centers)
            example['point_cls_labels'] = targets_dict['point_cls_labels']

        return example
