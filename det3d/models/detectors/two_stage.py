from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
import torch 
from torch import nn 

@DETECTORS.register_module
class TwoStageDetector(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        roi_head,
        point_head=None,
        num_point=1,
        freeze=False,
        **kwargs
    ):
        super(TwoStageDetector, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
        self.bbox_head = self.single_det.bbox_head

        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        for module in second_stage_modules:
            self.second_stage.append(builder.build_second_stage_module(module))

        self.roi_head = builder.build_roi_head(roi_head)

        self.point_head = builder.build_point_head(point_head) if point_head is not None else None

        self.num_point = num_point

    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    def get_box_center(self, boxes):
        centers = [] 
        for box in boxes:            
            if self.num_point == 1 or len(box['box3d_lidar']) == 0:
                centers.append(box['box3d_lidar'][:, :3])
            elif self.num_point == 5:
                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                corners = box_torch_ops.center_to_corner_box2d(center2d, dim2d, rotation_y)

                # front_middle = torch.cat([(corners[:, 0] + corners[:, 1])/2, height], dim=-1)
                # back_middle = torch.cat([(corners[:, 2] + corners[:, 3])/2, height], dim=-1)
                # left_middle = torch.cat([(corners[:, 0] + corners[:, 3])/2, height], dim=-1)
                # right_middle = torch.cat([(corners[:, 1] + corners[:, 2])/2, height], dim=-1)

                front_middle = torch.cat([(corners[:, 0] + corners[:, 1]) / 2], dim=-1)
                back_middle = torch.cat([(corners[:, 2] + corners[:, 3]) / 2], dim=-1)
                left_middle = torch.cat([(corners[:, 0] + corners[:, 3]) / 2], dim=-1)
                right_middle = torch.cat([(corners[:, 1] + corners[:, 2]) / 2], dim=-1)

                points = torch.cat([box['box3d_lidar'][:, :2], front_middle, back_middle, left_middle, \
                                    right_middle], dim=0)

                centers.append(points)
            elif isinstance(self.num_point, tuple):
                assert len(self.num_point) == 2
                grid_size = self.num_point

                center2d = box['box3d_lidar'][:, :2]
                height = box['box3d_lidar'][:, 2:3]
                dim2d = box['box3d_lidar'][:, 3:5]
                rotation_y = box['box3d_lidar'][:, -1]

                points = box_torch_ops.center_to_grid_box2d(center2d, dim2d, rotation_y, grid_size)
                #
                # box = box['box3d_lidar']
                # pp = torch.cat([points[0], height[:1].repeat(36, 1)], dim=1)
                # from tools.visual import draw_points, draw_boxes
                # viser = draw_points(pp.cpu().numpy(), point_size=10)
                # viser = draw_boxes(box[:1].cpu().numpy(), viser)
                # viser.run()

                centers.append(points)
            else:
                raise NotImplementedError()

        return centers

    def reorder_first_stage_pred_and_feature(self, first_pred, example):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1]

        if isinstance(self.num_point, tuple) or isinstance(self.num_point, list):
            num_point = self.num_point[0] * self.num_point[1]

        features = example['features']
        feature_vector_length = features[0].shape[-1] * num_point

        NMS_POST_MAXSIZE = self.single_det.test_cfg.nms.nms_post_max_size
        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, NMS_POST_MAXSIZE, box_length))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size, NMS_POST_MAXSIZE))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size, NMS_POST_MAXSIZE), dtype=torch.long)
        roi_features = features[0].new_zeros((batch_size, NMS_POST_MAXSIZE, feature_vector_length))

        for i in range(batch_size):
            num_obj = features[i].shape[0]
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target

            box_preds = first_pred[i]['box3d_lidar']

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            rois[i, :num_obj] = box_preds
            roi_labels[i, :500] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :500] = first_pred[i]['scores']
            roi_features[i, :num_obj] = features[i].view(num_obj, -1)

        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores  
        example['roi_features'] = roi_features

        example['has_class_labels']= True 

        return example 

    def post_process(self, batch_dict):
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1)) 
            mask = (label_preds != 0).reshape(-1)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            # currently don't need nms 
            pred_dict = {
                'box3d_lidar': box_preds,
                'scores': scores,
                'label_preds': labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 


    def forward(self, example, return_loss=True, **kwargs):
        out = self.single_det.forward_two_stage(example, return_loss, **kwargs)
        if len(out) == 4:
            one_stage_pred, bev_feature, voxel_feature, one_stage_loss = out
            example['bev_feature'] = bev_feature
            example['bone_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        one_stage_pred[0]['box3d_lidar'] = example['gt_box'][0][0]
        centers_vehicle_frame = self.get_box_center(one_stage_pred)

        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        for module in self.second_stage:
            example = module.forward(example, centers_vehicle_frame, self.num_point)

        example = self.reorder_first_stage_pred_and_feature(first_pred=one_stage_pred, example=example)

        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)
        if self.point_head is not None:
            batch_dict = self.point_head(example, centers_vehicle_frame, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict)
