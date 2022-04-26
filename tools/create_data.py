import copy
from pathlib import Path
import argparse

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.utils.create_gt_database_v1 import create_groundtruth_database as create_groundtruth_database_v1
from det3d.datasets.waymo import waymo_common as waymo_ds

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, trainval=False):
    # nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero, trainval_flag=trainval)
    if version == 'v1.0-trainval':
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    # waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("dataset", help="choose dataset function")
    parser.add_argument("nsweeps", type=int, help="only for waymo")
    parser.add_argument("split", type=str, default="val", help="nuScenes waymo")
    parser.add_argument("--trainval", action="store_true", help="only for nuscenes")
    args = parser.parse_args()

    if args.dataset == "waymo":
        assert args.nsweeps == 1 or args.nsweeps == 2
        print(args.nsweeps)
        waymo_data_prep("data/Waymo", split=args.split, nsweeps=int(args.nsweeps))
    elif args.dataset == "nuscenes":
        if args.split in ["train", "val"]:
            nuscenes_data_prep("data/nuScenes", version="v1.0-trainval", nsweeps=10, trainval=args.trainval)
        else:
            nuscenes_data_prep("data/nuScenes/v1.0-test", version="v1.0-test", nsweeps=10)
    else:
        raise NotImplementedError