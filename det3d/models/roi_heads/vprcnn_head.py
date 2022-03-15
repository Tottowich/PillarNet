import torch, torch.nn as nn
from det3d.ops.gnn_ops import gnn_modules as gnn_stack_modules
from .roi_head_template import RoIHeadTemplate
from ..registry import ROI_HEAD
from det3d.core.bbox.box_torch_ops import rotate_points_along_z


@ROI_HEAD.register_module
class VPRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, code_size=7, test_cfg=None):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.test_cfg = test_cfg
        self.code_size = code_size

        if self.model_cfg.ROI_GRID_POOL.TYPE == 'CORNER':
            self.model_cfg.ROI_GRID_POOL.GRID_SIZE = 3

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = gnn_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        if self.model_cfg.ROI_GRID_POOL.CENTER_AUG: c_out += 3
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)
    
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['relative_rois']
        xyz = batch_dict['xyz']  # in relative coordinates
        xyz_batch_cnt = batch_dict['xyz_batch_cnt']
        point_features = batch_dict['point_features']

        if 'point_cls_scores' in batch_dict.keys():
            point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        grid_sizes = global_roi_grid_points.shape[-2]
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, grid_sizes, 3)  # (B, Nx6x6x6, 3)

        _, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=global_roi_grid_points,
            rois=rois,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        if self.model_cfg.ROI_GRID_POOL.CENTER_AUG:
            roi_center = rois[:, :, :3].repeat_interleave(grid_sizes, dim=1)
            roi_center = global_roi_grid_points.view(-1, 3) - roi_center.view(-1, 3)
            pooled_features = torch.cat([roi_center, pooled_features], dim=1)

        pooled_features = pooled_features.view(-1, grid_sizes, pooled_features.shape[-1])  # (BxN, 6x6x6, C)
        return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])

        if self.model_cfg.ROI_GRID_POOL.TYPE == 'CORNER':
            local_roi_grid_points = self.get_dense_corner_points(rois)  # (B, 6x6x6, 3)
        else:
            local_roi_grid_points = self.get_dense_grid_points(rois, grid_size)  # (B, 6x6x6, 3)

        global_roi_grid_points = rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)

        # from tools.visual_utils.visualize_utils import draw_sphere_pts, boxes_to_corners_3d, draw_corners3d
        # import mayavi.mlab as mlab
        # fig = draw_sphere_pts(global_roi_grid_points[0], color=(0., 0., 1.), scale_factor=0.12)
        # corners = boxes_to_corners_3d(rois[:1])
        # fig = draw_corners3d(corners.cpu().numpy(), fig, color=(0., 1., 0.))
        # mlab.show(stop=True)

        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, grid_size):
        batch_size_rcnn = rois.shape[0]

        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    @staticmethod
    def get_dense_corner_points(rois):
        batch_size_rcnn = rois.shape[0]
        grid_size = 3
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        local_roi_size = local_roi_size.unsqueeze(dim=1)

        roi_corner_points = dense_idx / (grid_size - 1) * local_roi_size - (local_roi_size / 2)  # (B, 6x6x6, 3)
        return roi_corner_points

    def forward(self, batch_dict, training=True):
        if training:
            targets_dict = batch_dict.pop('targets_dict')

        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 3x3x3, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).contiguous().view(batch_size_rcnn, -1, 1)  # (BxN, C, 6x6x6)

        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
