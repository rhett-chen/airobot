import numpy as np
import cv2
import os
from options import make_parser
from util import *


def main():
    args = make_parser().parse_args()
    image = cv2.imread(os.path.join(args.data_path, 'rgb.jpg'))
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    pc = np.load(os.path.join(args.data_path, 'pc.npy'))  # scene point cloud
    print('load %s successfully' % os.path.join(args.data_path, 'pc.npy'))
    pc_colors = image.copy()
    pc_colors = np.reshape(pc_colors, [-1, 3])

    # object pos must equal to object pose in data_x/info.txt because point cloud and image are specified already
    if args.method == 'GPNet':
        grasp_poses, grasp_scores = load_pose_GPNet(os.path.join(args.data_path, args.pose_file))
    elif args.method == '6dof-graspnet':
        grasp_poses, grasp_scores = load_pose_6dofgraspnet(os.path.join(args.data_path, args.pose_file))
    elif args.method == 'graspnet_baseline':
        grasp_poses, grasp_scores = load_pose_graspnet_baseline(os.path.join(args.data_path, args.pose_file))
    else:
        raise NotImplementedError
    print('load %s successfully' % os.path.join(args.data_path, args.pose_file))

    mlab.figure(bgcolor=(1, 1, 1))
    draw_scene_mayavi(
        args,
        pc,
        pc_color=pc_colors,
        grasps=grasp_poses[:80],
        grasp_scores=grasp_scores[:80],
    )
    mlab.show()


if __name__ == '__main__':
    main()
