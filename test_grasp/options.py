import argparse


def make_parser():
    # Parameters needed for data acquisition, grasp, create robot, grasp visualization, etc
    # for the value that type is list, you should change default value instead of specify value on command line
    parser = argparse.ArgumentParser(description='Arguments for the whole grasp process',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--multi_obj', type=bool, default=True,
                        help='whether to load multi objects in scene, if true, object_pos/ori will be meaningless, '
                             'you need to change load_multi_objects function in util.py to customize the scene')
    parser.add_argument('--object_pos', type=list, default=[0.3, 0, 0.1],
                        help='for single object scene')
    parser.add_argument('--object_ori', type=list, default=[0, 0, 0.3],
                        help='for single object scene, in Euler angle, you need to set object type and size'
                             'in load_single_object function in util.py')

    parser.add_argument('--cam_focus_pt', type=list, default=[0.3, 0, 0.02],
                        help='camera focus point relative to the plane of robot in Pybullet simulation')
    parser.add_argument('--cam_pos', type=list, default=[0.15, 0, 0.2], help='camera position')
    parser.add_argument('--cam_height', type=int, default=180, help='camera resolution, height * width')
    parser.add_argument('--cam_width', type=int, default=180, help='camera resolution, height * width')

    parser.add_argument('--data_path', type=str, default='data_3', help='dir path that stores all relevant data')
    parser.add_argument('--pose_file', type=str, default='grasp_pose_baseline_obj_manual_seg.npy',
                        help='file in data_path dir that stores the grasp pose, .txt or .npy')

    parser.add_argument('--robot_arm', choices={'yumi_r', 'yumi_l', 'ur5e', 'franka'}, default='ur5e',
                        help='yumi has two arm, ur5e has one arm, this argument is used to specify which arm to '
                             'grasp: yumi_r | yumi_l | ur5e | franka')
    parser.add_argument('--control_mode', choices={"linear", "direct"}, default='linear',
                        help="linear | direct, 'direct' means to compute IK of target pose directly, 'linear' means to "
                             "set gripper to target orientation first, then move the gripper across the line from "
                             "current position to target position ans keep the orientation unchanged")

    parser.add_argument('--method', choices={"6dof-graspnet", "GPNet", "graspnet_baseline"},
                        default='graspnet_baseline',
                        help='grasp methods have been implemented: GPNet | 6dof-graspnet | graspnet_baseline')
    parser.add_argument('--vis_method', choices={'mayavi', 'open3d'}, default='mayavi',
                        help='use mayavi or open3d to visualize point cloud and grasps')
    return parser
