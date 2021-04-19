import numpy as np
import cv2
import os
from options import make_parser
from util import *


def main():
    args = make_parser().parse_args()
    pc = np.load(os.path.join(args.data_path, 'pc.npy'))  # scene point cloud
    print('load %s successfully' % os.path.join(args.data_path, 'pc.npy'))
    if args.method == 'GPNet':
        grasp_poses, grasp_scores = load_pose_GPNet(os.path.join(args.data_path, args.pose_file))
    elif args.method == '6dof-graspnet':
        grasp_poses, grasp_scores = load_pose_6dofgraspnet(os.path.join(args.data_path, args.pose_file))
    elif args.method == 'graspnet_baseline':
        grasp_poses, grasp_scores = load_pose_graspnet_baseline(os.path.join(args.data_path, args.pose_file))
    else:
        raise NotImplementedError
    print('load %s successfully' % os.path.join(args.data_path, args.pose_file))

    # if grasp_poses is None, it means you will visualize the point cloud data
    if args.vis_method == 'mayavi':
        image = cv2.imread(os.path.join(args.data_path, 'rgb.jpg'))
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        pc_colors = image.copy()
        pc_colors = np.reshape(pc_colors, [-1, 3])
        mlab.figure(bgcolor=(1, 1, 1))
        draw_scene_mayavi(
            pc=pc,
            pc_color=pc_colors,
            grasps=grasp_poses,
            grasp_scores=grasp_scores,
        )
        mlab.show()
    elif args.vis_method == 'open3d':
        from PIL import Image
        import open3d as o3d
        from graspnetAPI import GraspGroup

        color = np.array(Image.open(os.path.join(args.data_path, 'rgb.jpg')), dtype=np.float32) / 255.0
        color = color.reshape((-1, 3))
        gg_array = []
        offset_along_approach_vec = -0.02
        for index, pose in enumerate(grasp_poses):
            center = pose[0] + pos_offset_along_approach_vec(pose[2], offset_along_approach_vec)
            rot = ut.to_rot_mat(pose[1])
            te_1 = ut.euler2rot(np.array([0, -np.pi / 2, 0]))
            te_2 = ut.euler2rot(np.array([0, 0, np.pi / 2]))
            final_rot = np.dot(np.dot(rot, te_2), te_1)
            # grasp pose format is recorded in https://graspnetapi.readthedocs.io/en/latest/grasp_format.html
            te = [1, 0.065, 0.02, 0.02] if grasp_scores is None else [grasp_scores[index], 0.1, 0.02, 0.02]
            te = te + list(final_rot.flatten()) + list(center) + [-1]
            gg_array.append(te)
        gg = GraspGroup(np.array(gg_array))
        gg.nms()
        gg.sort_by_score()  # red for high score, blue for low
        max_grasps = 30
        if gg.__len__() > max_grasps:
            print('too many grasps, only keep {} grasps randomly!!!'.format(max_grasps))
            gg = gg.random_sample(max_grasps)
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
