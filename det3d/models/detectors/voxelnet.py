from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        dense_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, dense_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        data = self.reader(data)
        bev_feature, voxel_feature = self.backbone(
            data["features"], data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            bev_feature = self.neck(bev_feature)

        return bev_feature, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=example["points"]
        )

        bev_feature, _ = self.extract_feat(data)
        preds = self.bbox_head(bev_feature)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=example["points"]
        )

        bev_feature, voxel_feature = self.extract_feat(data)
        preds = self.bbox_head(bev_feature)

        example["points"] = data["points"]

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        # update test cfg
        self.test_cfg.score_threshold = self.test_cfg.second.score_threshold
        if return_loss:
            for k, v in self.test_cfg.second.nms_train.items():
                self.test_cfg.nms[k] = v
        else:
            for k, v in self.test_cfg.second.nms_test.items():
                self.test_cfg.nms[k] = v

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, voxel_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, voxel_feature, None 
