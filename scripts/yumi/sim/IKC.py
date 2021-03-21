import airobot as ar
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pybullet as p
from PIL import Image
from airobot.utils.common import euler2quat


def load_objects(robot):
    ori = euler2quat([0, 0, np.pi / 2])
    robot.pb_client.load_urdf('table/table.urdf',
                              [0.46, 0, 0],
                              ori,
                              scaling=0.9)
    # sphere_id = robot.pb_client.load_geom('sphere',
    #                                       size=0.05,
    #                                       mass=1,
    #                                       base_pos=[0.5, 0, 0.55],
    #                                       rgba=[0, 1, 0, 1])
    box_1_id = robot.pb_client.load_geom('box',
                                         size=0.03,
                                         mass=1,
                                         base_pos=[0.35, -0.23, 0.55],
                                         rgba=[1, 0, 0, 1])
    box_2_id = robot.pb_client.load_geom('box',
                                         size=[0.03, 0.012, 0.015],
                                         mass=1,
                                         base_pos=[0.53, 0.17, 0.55],
                                         rgba=[0, 0, 1, 1])
    cylinder_id = robot.pb_client.load_geom('cylinder',
                                            size=[0.03, 0.04],
                                            mass=1,
                                            base_pos=[0.49, -0.12, 0.55],
                                            rgba=[0, 1, 1, 1])
    duck_id = robot.pb_client.load_geom('mesh',
                                        mass=1,
                                        visualfile='duck.obj',
                                        mesh_scale=0.06,
                                        base_pos=[0.63, 0.3, 0.55],
                                        rgba=[0.5, 0.2, 1, 1])


def main():
    """
    Move the robot end effector to the desired pose.
    """
    robot = ar.Robot('yumi_grippers')
    robot.arm.go_home()
    robot.pb_client.load_urdf('plane.urdf')
    load_objects(robot)
    robot_z_offset = 0.58
    robot.cam.setup_camera(focus_pt=[0, 0, 0.25 + robot_z_offset], camera_pos=[1.4, 0, 0.3 + robot_z_offset])
    # robot.cam.setup_camera(focus_pt=[0, 0, 0.5], dist=1.5, height=800, width=800)

    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rgb_img_dir = os.path.join(root_path, 'rgb')
    rgb_index = 0
    pos, quat, rot, euler = robot.arm.get_ee_pose(arm='right')
    ar.log_info('Right arm end effector position: ' + str(pos) + ' ' + str(euler))
    # initial right end effector pos: [ 0.3196905  -0.53293526  0.27487105] [3.08614804 -0.56711174  0.58926922]

    pos_delta = 0.3
    euler_delta = 0.2
    open_flag = False
    target_pos, target_ori = [], []
    while 1:
        mov = input()
        if mov == 'w':
            target_pos = [pos[0] - pos_delta, pos[1], pos[2]]
        elif mov == 'a':
            target_pos = [pos[0], pos[1] - pos_delta, pos[2]]
        elif mov == 's':
            target_pos = [pos[0] + pos_delta, pos[1], pos[2]]
        elif mov == 'd':
            target_pos = [pos[0], pos[1] + pos_delta, pos[2]]
        elif mov == 'r':
            target_pos = [pos[0], pos[1], pos[2] + pos_delta]
        elif mov == 'f':
            target_pos = [pos[0], pos[1], pos[2] - pos_delta]
        elif mov == 'z':
            target_ori = [euler[0] - euler_delta, euler[1], euler[2]]
        elif mov == 'x':
            target_ori = [euler[0] + euler_delta, euler[1], euler[2]]
        elif mov == 't':
            target_ori = [euler[0], euler[1] - euler_delta, euler[2]]
        elif mov == 'g':
            target_ori = [euler[0], euler[1] + euler_delta, euler[2]]
        elif mov == 'c':
            target_ori = [euler[0], euler[1], euler[2] - euler_delta]
        elif mov == 'v':
            target_ori = [euler[0], euler[1], euler[2] + euler_delta]
        elif mov == ' ':
            open_flag = not open_flag
            if open_flag:
                robot.arm.right_arm.eetool.open()
            else:
                robot.arm.right_arm.eetool.close()
            continue
        elif mov == 'q':
            img, depth, seg = robot.cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            # scale = 25.
            # sdepth = depth * scale
            cv2.imwrite(os.path.join(rgb_img_dir, str(rgb_index).zfill(5) + '_rgb.jpg'), img)
            # cv2.imwrite(os.path.join(rgb_img_dir, str(rgb_index).zfill(5) + '_depth.jpg'), sdepth.astype(np.uint16))
            # cv2.imwrite(os.path.join(rgb_img_dir, str(rgb_index).zfill(5) + '_seg.jpg'), seg)
            # Image.fromarray(seg).save(os.path.join(rgb_img_dir, str(rgb_index).zfill(5) + '_seg.png'))
            ar.log_info('Successfully saved rgb image ' + rgb_img_dir + '\\' + str(rgb_index).zfill(5) + '.jpg')
            rgb_index += 1
            continue
        else:
            robot.arm.go_home()
            pos, quat, rot, euler = robot.arm.get_ee_pose(arm='right')
            ar.log_info('Right arm end effector position: ' + str(pos))
            continue
        if mov in ['w', 'a', 's', 'd', 'r', 'f']:
            robot.arm.set_ee_pose(target_pos, arm='right')
        else:
            robot.arm.set_ee_pose(ori=target_ori, arm='right')
        pos, quat, rot, euler = robot.arm.get_ee_pose(arm='right')
        ar.log_info('Right arm end effector position: ' + str(pos) + ' ' + str(euler))


if __name__ == '__main__':
    main()
