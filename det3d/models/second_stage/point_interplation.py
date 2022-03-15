import numpy as np
import torch
from torch import nn

from det3d.models.roi_heads.roi_head_template import RoIHeadTemplate
from det3d.models.registry import SECOND_STAGE
from det3d.core.utils.center_utils import (
    bilinear_interpolate_torch,
)

from det3d.ops.points_ops.points_utils import furthest_point_downsample, points_in_rois
from det3d.ops.pillar_ops.pillar_modules import SparseInterpolate3D
from det3d.models.utils import build_norm_layer


@SECOND_STAGE.register_module
class PointFeatureExtractor(RoIHeadTemplate):
    def __init__(self, pc_range, voxel_size, out_channels, model_cfg):
        super().__init__(num_class=1, model_cfg=model_cfg)
        self.pc_range = pc_range
        self.voxel_size = np.array(voxel_size, dtype=np.float32)

        self.PF_layers = nn.ModuleList()
        self.PF_layer_names = []
        self.PF_strides = {}

        c_in = 0
        for conv, p_cfg in model_cfg.PF_LAYER.items():
            c_in += p_cfg.CHN
            self.PF_layer_names.append(conv)
            self.PF_strides[conv] = p_cfg.STRIDE
            if conv in ['bev']:
                continue
            self.PF_layers.append(SparseInterpolate3D(p_cfg.KERNEL,
                                                      self.voxel_size * p_cfg.STRIDE))

        self.point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, out_channels, bias=False),
            build_norm_layer(dict(type="BN1d", eps=1e-3, momentum=0.01), out_channels)[1],
            nn.ReLU()
        )

    @torch.no_grad()
    def interpolate_from_bev_features(self, points_list, bev_features, bev_stride):

        point_bev_feature = []
        for k, points in enumerate(points_list):
            point_bev_feature.append(bilinear_interpolate_torch(bev_features[k],
                                     points[:, 0] / (self.voxel_size[0] * bev_stride),
                                     points[:, 1] / (self.voxel_size[1] * bev_stride)))

        return torch.cat(point_bev_feature, dim=0)  # (M1+M2 ..., C)

    @torch.no_grad()
    def get_attendant_points(self, batch_rois, batch_points):
        batch_size = len(batch_rois)

        start = 0
        attendant_points = []
        for k in range(batch_size):
            points = points_in_rois(batch_rois[k].contiguous(),
                                    batch_points[k].contiguous(), self.model_cfg.EXTRA)
            # from tools.visual import draw_points, draw_boxes
            # viser = draw_points(points.cpu().numpy())
            # viser = draw_boxes(batch_rois[k].cpu().numpy(), viser)
            # viser.run()
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                if len(points) > self.model_cfg.MAX_POINTS:
                    points = furthest_point_downsample(self.model_cfg.MAX_POINTS, points.contiguous())
            # draw_sphere_pts(points, (1, 1, 0), fig=fig)
            # mlab.show(stop=True)
            attendant_points.append(points[:, :3])

        return attendant_points

    @torch.no_grad()
    def absl_to_relative(self, absolute):
        relative = absolute.detach().clone()
        relative[..., 0] -= self.pc_range[0]
        relative[..., 1] -= self.pc_range[1]
        relative[..., 2] -= self.pc_range[2]

        return relative

    @torch.no_grad()
    def relative_to_coords(self, relative, xyz_batch_cnt):
        absl = relative.clone()
        absl[..., 0] += self.pc_range[0]
        absl[..., 1] += self.pc_range[1]
        absl[..., 2] += self.pc_range[2]

        point_coords = []

        start = 0
        for b, num in enumerate(xyz_batch_cnt):
            points = absl[start:num.item(), :]
            point_coords.append(torch.cat((torch.full((len(points), 1), b, dtype=absl.dtype, device=absl.device),
                                           points), dim=1))

        point_coords = torch.cat(point_coords, dim=0)
        return point_coords.contiguous()

    def forward(self, example, training=True):
        if training:
            targets_dict = self.assign_targets(example)
            example['rois'] = targets_dict['rois']
            example['roi_labels'] = targets_dict['roi_labels']
            example['targets_dict'] = targets_dict

        # relative positions to minimum range
        relative_points = [self.absl_to_relative(points) for points in example['points']]
        example['relative_rois'] = self.absl_to_relative(example['rois'])

        relative_points = self.get_attendant_points(example['relative_rois'], relative_points)
        xyz_batch_cnt = torch.IntTensor([len(pts) for pts in relative_points]).to(example['rois'].device)

        point_features = []
        if 'bev' in self.PF_layer_names:
            point_features.append(self.interpolate_from_bev_features(relative_points,
                                                                     example['bev_feature'],
                                                                     self.PF_strides['bev']))
        relative_points = torch.cat(relative_points, dim=0)
        point_features.append(relative_points[:, 3:])
        relative_points = relative_points[:, :3].contiguous()

        cnt = 0
        for src_name in self.PF_layer_names:
            if src_name not in ['conv1', 'conv2', 'conv3', 'conv4']: continue
            point_features.append(self.PF_layers[cnt](
                example['voxel_feature'][src_name],
                xyz=relative_points,
                xyz_batch_cnt=xyz_batch_cnt,
            ))
            cnt += 1

        example['point_features'] = self.point_feature_fusion(torch.cat(point_features, dim=1))
        # example['point_coords'] = self.relative_to_coords(relative_points, xyz_batch_cnt)
        example['xyz'] = relative_points
        example['xyz_batch_cnt'] = xyz_batch_cnt

        return example
