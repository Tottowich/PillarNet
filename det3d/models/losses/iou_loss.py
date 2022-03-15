import torch


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
                                dtype=torch.float32, device=dim.device)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.min(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.max(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.max(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.min(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_qboxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gboxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.min(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.max(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gboxes + volume_qboxes - volume_inter

    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.max(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.min(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h ** 2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious


def bbox3d_overlaps_giou(pred_boxes, gt_boxes):
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = torch.min(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.max(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.max(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.min(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_qboxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gboxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.min(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.max(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gboxes + volume_qboxes - volume_inter

    outer_h = torch.max(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              torch.min(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    closure = outer[:, 0] * outer[:, 1] * outer_h

    gious = volume_inter / volume_union - (closure - volume_union) / closure
    gious = torch.clamp(gious, min=-1.0, max=1.0)
    return gious