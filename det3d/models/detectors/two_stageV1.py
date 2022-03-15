from det3d.core.bbox import box_torch_ops
from ..registry import DETECTORS
from .base import BaseDetector
from .. import builder
from det3d.torchie.trainer import load_checkpoint
import torch 
from torch import nn

@DETECTORS.register_module
class TwoStageDetectorV1(BaseDetector):
    def __init__(
        self,
        first_stage_cfg,
        second_stage_modules,
        roi_head, 
        num_point=1,
        freeze=False,
        pretrained=None,
        **kwargs
    ):
        super(TwoStageDetectorV1, self).__init__()
        self.single_det = builder.build_detector(first_stage_cfg, **kwargs)

        if freeze:
            print("Freeze First Stage Network")
            # we train the model in two steps 
            self.single_det = self.single_det.freeze()
        self.bbox_head = self.single_det.bbox_head

        self.second_stage = nn.ModuleList()
        # can be any number of modules 
        # bird eye view, cylindrical view, image, multiple timesteps, etc.. 
        # for module in second_stage_modules:
        #     self.second_stage.append(builder.build_second_stage_module(module))
        self.second_stage = builder.build_second_stage_module(second_stage_modules)

        self.roi_head = builder.build_roi_head(roi_head)

        self.num_point = num_point

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def combine_loss(self, one_stage_loss, roi_loss, tb_dict):
        one_stage_loss['loss'][0] += (roi_loss)

        for i in range(len(one_stage_loss['loss'])):
            one_stage_loss['roi_reg_loss'].append(tb_dict['rcnn_loss_reg'])
            one_stage_loss['roi_cls_loss'].append(tb_dict['rcnn_loss_cls'])

        return one_stage_loss

    def reorder_first_stage_pred(self, first_pred, example):
        batch_size = len(first_pred)
        box_length = first_pred[0]['box3d_lidar'].shape[1]

        NMS_POST_MAXSIZE = self.single_det.test_cfg.nms.nms_post_max_size
        rois = first_pred[0]['box3d_lidar'].new_zeros((batch_size, NMS_POST_MAXSIZE, box_length))
        roi_scores = first_pred[0]['scores'].new_zeros((batch_size, NMS_POST_MAXSIZE))
        roi_labels = first_pred[0]['label_preds'].new_zeros((batch_size, NMS_POST_MAXSIZE), dtype=torch.long)

        for i in range(batch_size):
            # basically move rotation to position 6, so now the box is 7 + C . C is 2 for nuscenes to
            # include velocity target
            box_preds = first_pred[i]['box3d_lidar']

            if self.roi_head.code_size == 9:
                # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 8, 6, 7]]

            num_obj = min(NMS_POST_MAXSIZE, len(box_preds))
            rois[i, :num_obj] = box_preds
            roi_labels[i, :num_obj] = first_pred[i]['label_preds'] + 1
            roi_scores[i, :num_obj] = first_pred[i]['scores']

        example['batch_size'] = batch_size
        example['rois'] = rois 
        example['roi_labels'] = roi_labels 
        example['roi_scores'] = roi_scores
        example['has_class_labels'] = True

        return example 

    def post_process(self, batch_dict, test_cfg):
        batch_size = batch_dict['batch_size']
        pred_dicts = [] 

        for index in range(batch_size):
            box_preds = batch_dict['batch_box_preds'][index]
            cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou 
            label_preds = batch_dict['roi_labels'][index]

            if box_preds.shape[-1] == 9:
                # move rotation to the end (the create submission file will take elements from 0:6 and -1) 
                box_preds = box_preds[:, [0, 1, 2, 3, 4, 5, 7, 8, 6]]

            # scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
            scores = torch.sigmoid(cls_preds).reshape(-1)
            mask = (label_preds != 0).reshape(-1) & (scores > test_cfg.score_threshold)

            box_preds = box_preds[mask, :]
            scores = scores[mask]
            labels = label_preds[mask]-1

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]
            selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms, scores,
                                                      thresh=test_cfg.nms.nms_iou_threshold,
                                                      pre_maxsize=test_cfg.nms.nms_pre_max_size,
                                                      post_max_size=test_cfg.nms.nms_post_max_size)

            selected_boxes = box_preds[selected]
            selected_scores = scores[selected]
            selected_labels = labels[selected]
            pred_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels,
                "metadata": batch_dict["metadata"][index]
            }

            pred_dicts.append(pred_dict)

        return pred_dicts 


    def forward(self, example, return_loss=True, **kwargs):
        out = self.single_det.forward_two_stage(example, 
            return_loss, **kwargs)
        if len(out) == 4:
            one_stage_pred, bev_feature, voxel_feature, one_stage_loss = out 
            example['voxel_feature'] = voxel_feature
        elif len(out) == 3:
            one_stage_pred, bev_feature, one_stage_loss = out 
        else:
            raise NotImplementedError

        # N C H W -> N H W C 
        example['bev_feature'] = bev_feature.permute(0, 2, 3, 1).contiguous()

        if self.roi_head.code_size == 7 and return_loss is True:
            # drop velocity 
            example['gt_boxes_and_cls'] = example['gt_boxes_and_cls'][:, :, [0, 1, 2, 3, 4, 5, 6, -1]]

        example = self.reorder_first_stage_pred(first_pred=one_stage_pred, example=example)
        example = self.second_stage.forward(example, training=return_loss)

        # final classification / regression 
        batch_dict = self.roi_head(example, training=return_loss)

        if return_loss:
            roi_loss, tb_dict = self.roi_head.get_loss()

            return self.combine_loss(one_stage_loss, roi_loss, tb_dict)
        else:
            return self.post_process(batch_dict, self.single_det.test_cfg)
