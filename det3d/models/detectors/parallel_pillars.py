from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class ParallelPillars(SingleStageDetector):
    def __init__(
        self,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(ParallelPillars, self).__init__(
            None, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        x = self.neck(
            data["xyz"], data["xyz_batch_cnt"], data["point_features"]
        )

        return x

    def forward(self, example, return_loss=True, **kwargs):
        batch_size = len(example['metadata'])

        data = dict(
            xyz=example["xyz"],
            point_features=example["point_features"],
            xyz_batch_cnt=example["xyz_batch_cnt"],
            batch_size=batch_size,
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
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
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds)
        else:
            return boxes, bev_feature, None 