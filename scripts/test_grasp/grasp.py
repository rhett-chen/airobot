import sys
sys.path.append('airobot/')
import airobot as ar
import numpy as np
import os
import time
from get_data import load_single_object
import airobot.utils.common as ut
from util import *
from options import make_parser


def manual_control(args):
    open_flag = False
    robot_control_type = 'ur5e' if args.robot_arm == 'franka' else args.robot_arm
    pos, quat, rot, euler = dispatch_control_order(robot_control_type + ':get_pose')
    ar.log_info('Right arm end effector position: ' + str(pos) + ' ' + str(euler))
    while 1:
        mov = input()
        if mov == ' ':
            # open or close the gripper
            open_flag = not open_flag
            if open_flag:
                dispatch_control_order(robot_control_type + ':open')
            else:
                dispatch_control_order(robot_control_type + ':close')
        elif mov == 'pose':
            # input the pos and ori, then control robot to the target pose
            temp = input().strip()  # example input: 0 0 0
            print(temp + '\n')
            center = np.array([float(p) for p in temp.split()])
            temp = input().strip()
            print(temp + '\n')
            quaternion = [float(p) for p in temp.split()]
            print('center: {}, ori: {}'.format(center, quaternion))
            dispatch_control_order(robot_control_type + ':set_pose', center, quaternion)
        elif mov == 'home':
            dispatch_control_order(robot_control_type + ':home')
        elif mov == 'dp':
            # input the delta of pos, then control robot
            dx = float(input().strip())
            dy = float(input().strip())
            dz = float(input().strip())
            target_pos = [pos[0] + dx, pos[1] + dy, pos[2] + dz]
            print('target pos: ', target_pos)
            dispatch_control_order(robot_control_type + ':set_pose', target_pos)
        elif mov == 'do':
            # input the delta of ori in euler angels, then control robot
            dx = float(input().strip())
            dy = float(input().strip())
            dz = float(input().strip())
            target_ori = [euler[0] + dx, euler[1] + dy, euler[2] + dz]
            print('target ori: ', target_ori)
            dispatch_control_order(robot_control_type + ':set_pose', pos=None, ori=target_ori)
        else:
            pass
        pos, quat, rot, euler = dispatch_control_order(robot_control_type + ':get_pose')
        ar.log_info('Right arm end effector position: ' + str(pos) + ' ' + str(euler) + ' ' + str(quat))


def dispatch_control_order(order, pos=None, ori=None):
    return {
        'yumi_r:open': lambda: robot.arm.right_arm.eetool.open(),
        'yumi_r:close': lambda: robot.arm.right_arm.eetool.close(),
        'yumi_l:open': lambda: robot.arm.left_arm.eetool.open(),
        'yumi_l:close': lambda: robot.arm.left_arm.eetool.close(),
        'ur5e:open': lambda: robot.arm.eetool.open(),
        'ur5e:close': lambda: robot.arm.eetool.close(),
        'yumi_r:get_pose': lambda: robot.arm.get_ee_pose(arm='right'),
        'yumi_l:get_pose': lambda: robot.arm.get_ee_pose(arm='left'),
        'ur5e:get_pose': lambda: robot.arm.get_ee_pose(),
        'yumi_r:set_pose': lambda: robot.arm.set_ee_pose(pos=pos, ori=ori, arm='right'),
        'yumi_l:set_pose': lambda: robot.arm.set_ee_pose(pos=pos, ori=ori, arm='left'),
        'ur5e:set_pose': lambda: robot.arm.set_ee_pose(pos=pos, ori=ori),
        'yumi_r:move_xyz': lambda: robot.arm.move_ee_xyz(pos, eef_step=0.01, arm='right'),
        'yumi_l:move_xyz': lambda: robot.arm.move_ee_xyz(pos, eef_step=0.01, arm='left'),
        'ur5e:move_xyz': lambda: robot.arm.move_ee_xyz(pos, eef_step=0.01),
        'yumi_r:home': lambda: robot.arm.go_home(arm='right'),
        'yumi_l:home': lambda: robot.arm.go_home(arm='left'),
        'ur5e:home': lambda: robot.arm.go_home(),
    }.get(order, lambda: None)()


def control_robot(pose, robot_category='yumi_r', control_mode='direct', move_up=True, go_home=True,
                  linear_offset=-0.022):
    """
    Given the position and quaternion of target pose, choose the robot arm and control mode, then control the robot
    gripper to target pose
    Args:
        linear_offset: linear offset between gripper position and grasp position center
        pose: pose[0] position, list of size 3
              pose[1]: often be quaternion, list of size 4; also can be rotation matrix or euler angels
              pose[2[: approaching vector, list of size 3
        robot_category: 'yumi_l' means left arm of YUMI robot, so is 'yumi_r', ur5e means ur5e robot, franka...
        control_mode: 'direct' means compute IK of target pose directly, 'linear' means first set gripper to target
                      orientation, then move the gripper across the line from current position to target position ans
                      keep the orientation unchanged
        move_up: after closing the gripper, whether move gripper up to a certain height
        go_home: return to original state
    """
    if control_mode not in ['direct', 'linear']:
        raise NotImplementedError
    if robot_category not in ['yumi_r', 'yumi_l', 'ur5e', 'franka']:
        raise NotImplementedError
    if robot_category == 'franka':
        robot_category = 'ur5e'  # The control commands of these two robots are the same in this simulation environment

    # there is a linear offset between gripper position and grasp position center,
    actual_target_pos = pose[0] + pos_offset_along_ori(pose[2], linear_offset)
    print('target grasp pose: pos|quat|approach vector', actual_target_pos, pose[1], pose[2])

    dispatch_control_order(robot_category + ':open')
    if control_mode == 'direct':
        dispatch_control_order(robot_category + ':set_pose', actual_target_pos, pose[1])
    elif control_mode == 'linear':
        temp_posi = pose[0] + pos_offset_along_ori(pose[2], -0.25)
        dispatch_control_order(robot_category + ':set_pose', temp_posi, pose[1])
        cur_pos, cur_quat, _, cur_euler = dispatch_control_order(robot_category + ':get_pose')
        delta_pos = np.array(actual_target_pos) - np.array(cur_pos)
        dispatch_control_order(robot_category + ':move_xyz', delta_pos)
    cur_pos, cur_quat, _, cur_euler = dispatch_control_order(robot_category + ':get_pose')
    print('current (pos|quat|euler): ', cur_pos, cur_quat, cur_euler)
    dispatch_control_order(robot_category + ':close')
    time.sleep(1)
    if move_up:
        # up_target_pos = np.array(cur_pos) + np.array([0, 0, 0.13])
        up_target_pos = np.array([0.4, 0, 0.3])
        dispatch_control_order(robot_category + ':set_pose', up_target_pos)
        print('done!')
        time.sleep(2)
        dispatch_control_order(robot_category + ':open')
    if go_home:
        dispatch_control_order(robot_category + ':home')


def auto_control(args, obj_id=None):
    """
    Automatically read grasp pose from grasp pose file, this function only for grasp single object,
    and object type, position, orientation must be the same with data_x/info.txt
       Besides, set the position that input into control_robot() as the center point of contact point uniformly¡£
    Args:
        args: parser
        obj_id: pybullet object ID, used to reset the arena
    """
    # Set the position as the center point of contact point uniformly
    if args.method == 'GPNet':
        poses, scores = load_pose_GPNet(os.path.join(args.data_path, args.pose_file))
    elif args.method == '6dof-graspnet':
        poses, scores = load_pose_6dofgraspnet(os.path.join(args.data_path, args.pose_file))
        offset_along_ori = 0.10527  # 6dof-graspnet, pos is bottom point, change to center of contact points
        for center, _, vec in poses:
            center += pos_offset_along_ori(vec, offset_along_ori)
    else:
        raise NotImplementedError

    for pose in poses:
        control_robot(pose, robot_category=args.robot_arm, control_mode=args.control_mode, move_up=True)
        time.sleep(3)
        robot.pb_client.reset_body(obj_id, base_pos=args.object_pos, base_quat=ut.euler2quat(args.object_ori))
        time.sleep(0.8)
    print('all done!!!!')
    input()


if __name__ == '__main__':
    # when set orientation is euler angle [0, 0, 0], the gripper is vertical upward, the gripper plane is x-z plane
    args = make_parser().parse_args()
    if args.robot_arm == 'ur5e':
        robot_type = 'ur5e_2f140'
    elif args.robot_arm == 'franka':
        robot_type = 'franka'
    else:
        if args.robot_arm.split('_')[0] == 'yumi':
            robot_type = 'yumi_grippers'
        else:
            raise NotImplementedError("robot_arm can only be one of ['yumi_r', 'yumi_l', 'ur5e', 'franka']")
    robot = ar.Robot(robot_type)
    robot.arm.go_home()
    robot.pb_client.load_urdf('plane.urdf')
    if args.robot_z_offset > 0:
        robot.pb_client.load_urdf('table/table.urdf', base_pos=[0.1, 0, 0], scaling=0.9)
    box_id = load_single_object(robot, args)
    time.sleep(0.8)
    args.cam_focus_pt[2] += args.robot_z_offset
    args.cam_pos[2] += args.robot_z_offset
    robot.cam.setup_camera(focus_pt=args.cam_focus_pt, camera_pos=args.cam_pos,
                           height=args.cam_height, width=args.cam_width)

    # manual_control(args)  # Not recommend
    auto_control(args, obj_id=box_id)
