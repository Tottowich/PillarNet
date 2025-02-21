from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 


@DETECTORS.register_module
class PillarNet(SingleStageDetector):
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
        super(PillarNet, self).__init__(
            reader, backbone, neck, dense_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        data = self.reader(data)
        pillar_features = self.backbone(
            data["xyz"], data["xyz_batch_cnt"], data["pt_features"]
        )
        if self.with_neck:
            x = self.neck(pillar_features)

        return x, pillar_features

    def forward(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        bev_feature, bone_features  = self.extract_feat(data)
        preds = self.bbox_head(bev_feature)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        bev_feature, bone_features = self.extract_feat(data)
        preds = self.bbox_head(bev_feature, bone_features)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, bone_features, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, bone_features, None
