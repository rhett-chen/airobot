import argparse


def make_parser():
    # Parameters needed for data acquisition, grasp, create robot, grasp visualization, etc
    parser = argparse.ArgumentParser(description='Arguments for the whole grasp process',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--multi_obj', type=bool, default=True,
                        help='whether to load multi objects in scene, if true, object_pos/ori will be meaningless, '
                             'you need to change load_multi_objects function in util.py to customize the scene')
    parser.add_argument('--object_pos', type=list, default=[0.3, 0, 0.1], help='object position')
    parser.add_argument('--object_ori', type=list, default=[0, 0, 0.3],
                        help='object orientation in Euler angle, you need to set object type and size'
                             'in load_single_object function in util.py')

    parser.add_argument('--cam_focus_pt', type=list, default=[0.3, 0, 0.02],
                        help='camera focus point relative to the plane of robot in Pybullet simulation')
    parser.add_argument('--cam_pos', type=list, default=[0.15, 0, 0.2], help='camera position')
    parser.add_argument('--cam_height', type=int, default=180, help='camera resolution, height * width')
    parser.add_argument('--cam_width', type=int, default=180, help='camera resolution, height * width')

    parser.add_argument('--robot_z_offset', type=float, default=0.0,
                        help='offset of robot plane to [0, 0, 0], If the value is not 0, a table will be imported into '
                             'the scene. It is not recommended because manual parameters adjustment is required.'
                             'The relevant parameters are object pos, table pos(get_data.py 36 and grasp.py 182) ')

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
    return parser
