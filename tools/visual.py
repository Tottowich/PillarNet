import open3d as o3d
import argparse
import pickle
import numpy as np
from det3d.core.bbox.box_np_ops import center_to_corner_box3d

def label2color(label):
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255]]

    return colors[label]

def corners_to_lines(qs, color=[204/255, 0, 0]):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1,0), (5,4), (2,3), (6,7), (1,2), (5,6), (0,3), (4,7), (1,5), (0,4), (2,6), (3,7)]
    cl = [color for i in range(12)]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)
    
    return line_set

def plot_boxes(boxes, score_thresh):
    visuals =[] 
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue 

        box = boxes['boxes'][i:i+1]
        label = boxes['classes'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals

def draw_points(points, viser=None, color=(1, 0, 0), point_size=0.05):
    if viser is None:
        viser = o3d.visualization.Visualizer()
        viser.create_window()
    viser.get_render_option().point_size = point_size

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color(color)
    viser.add_geometry(pcd)
    viser.update_renderer()
    viser.poll_events()
    return viser

def draw_boxes(boxes, viser=None, color=(0, 1, 0)):
    if viser is None:
        viser = o3d.visualization.Visualizer()
        viser.create_window()

    corners = center_to_corner_box3d(boxes[:, :3], boxes[:, 3:6], boxes[:, -1])
    for corner in corners:
        viser.add_geometry(corners_to_lines(corner, color))
    viser.poll_events()
    return viser


if __name__ == '__main__':
    points = np.load('/home/sgs/points.npy')
    rois = np.load('/home/sgs/boxes.npy')
    viser = draw_points(points, color=(214/255, 93/255, 177/255), point_size=1)
    viser = draw_boxes(rois, viser)
    viser.run()
    viser.capture_screen_image("/home/sgs/example.png", do_render=True)
    # points = np.random.rand(1000, 3) * 10
    # viser = plot_points(points)
    # viser.run()
    # parser = argparse.ArgumentParser(description="CenterPoint")
    # parser.add_argument('--path', help='path to visualization file', type=str)
    # parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    # args = parser.parse_args()
    #
    # with open(args.path, 'rb') as f:
    #     data_dicts = pickle.load(f)
    #
    # for data in data_dicts:
    #     points = data['points']
    #     detections = data['detections']
    #
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    #
    #     visual = [pcd]
    #     num_dets = detections['scores'].shape[0]
    #     visual += plot_boxes(detections, args.thresh)
    #
    #     o3d.visualization.draw_geometries(visual)
