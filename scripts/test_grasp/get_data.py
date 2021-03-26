import sys
sys.path.append('../airobot/')
import airobot as ar
import numpy as np
import cv2
import os
import time
import pybullet as p
import numpy as np
from util import load_single_object
from options import make_parser


def encode_depth_to_image(dmap):
    min_v = np.min(dmap)
    max_v = np.max(dmap)
    # print('miv max v: ', min_v, max_v)
    v_range = max(1e-5, (max_v - min_v))
    dmap_norm = (dmap - min_v) / v_range
    dmap_norm = (dmap_norm * 2 ** 8).astype(np.uint8)
    dmap_norm[dmap == 0] = 255
    return dmap_norm


def main():
    """
      in current dir: test_grasp, we will save the whole info(depth, point cloud, info.npy, info.txt) to a
    separate folder, named data_x automatically, x is the number of folders in the current dir
    """
    # load robot and object, set camera parameter
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
    load_single_object(robot, args)
    time.sleep(1.3)
    args.cam_focus_pt[2] += args.robot_z_offset
    args.cam_pos[2] += args.robot_z_offset
    robot.cam.setup_camera(focus_pt=args.cam_focus_pt, camera_pos=args.cam_pos,
                           height=args.cam_height, width=args.cam_width)

    # in current dir: test_grasp, we will save the whole info(depth, point cloud, info.npy, info.txt) to a
    # separate folder, named data_x, x is the number of folders in the current dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_num = len([lists for lists in os.listdir(current_dir)
                      if os.path.isdir(os.path.join(current_dir, lists))]) - 1
    save_path = os.path.join(current_dir, 'data_' + str(folder_num))
    os.mkdir(save_path)
    # save camera info to camera_info.npy, in addition, save object and camera info to info.txt
    info = {'cam_intrinsic': robot.cam.cam_int_mat, 'cam_view': robot.cam.view_matrix, 'cam_external':
            robot.cam.cam_int_mat, 'cam_pos': args.cam_pos, 'cam_focus_pt': args.cam_focus_pt,
            'cam_height': args.cam_height, 'cam_width': args.cam_width, 'object_pos': args.object_pos,
            'obj_ori': args.object_ori}
    np.save(os.path.join(save_path, 'info.npy'), info)

    with open(os.path.join(save_path, 'info.txt'), 'w') as ci:
        ci.write('object position: \n')
        ci.write(str(args.object_pos) + '\n')
        ci.write('object rotation: \n')
        ci.write(str(args.object_ori) + '\n')
        ci.write('camera position: \n')
        ci.write(str(args.cam_pos) + '\n')
        ci.write('camera focus point: \n')
        ci.write(str(args.cam_focus_pt) + '\n')
        ci.write('camera resolution: \n')
        ci.write(str(args.cam_height) + ' * ' + str(args.cam_width) + '\n')
        ci.write('camera z offset: \n')
        ci.write(str(args.robot_z_offset) + '\n')
        ci.write('camera view matrix: \n')
        ci.write(str(robot.cam.view_matrix) + '\n')
        ci.write('camera intrinsic matrix: ' + '\n')
        ci.write(str(robot.cam.cam_int_mat) + '\n')
        ci.write('camera external matrix: ' + '\n')
        ci.write(str(robot.cam.cam_ext_mat))
        ci.flush()

    # save rgb. depth, point cloud
    img, depth, _ = robot.cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
    print('image shape: {}, depth shape: {}'.format(img.shape, depth.shape))
    cv2.imwrite(os.path.join(save_path, 'depth.jpg'),
                encode_depth_to_image(depth))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    cv2.imwrite(os.path.join(save_path, 'rgb.jpg'), img)
    ar.log_info(f'Depth image min:{depth.min()} m, max: {depth.max()} m.')
    np.save(os.path.join(save_path, 'depth.npy'), depth)

    # get point cloud data in the world frame
    pts = robot.cam.depth_to_point_cloud(depth, in_world=True)
    ar.log_info('point cloud shape: {}'.format(pts.shape))
    print('\npoint clouds: \n')
    print(pts.reshape([args.cam_height, args.cam_width, 3])[100, 100])
    np.save(os.path.join(save_path, 'pc.npy'), pts)
    print('saved %s successfully' % os.path.join(save_path, 'pc.npy'))
    pco = pts.reshape([args.cam_height, args.cam_width, 3])[50:130, 60:120, :]
    np.save(os.path.join(save_path, 'pco.npy'), pco.reshape((-1, 3)))  # part of point cloud
    print('saved %s successfully' % os.path.join(save_path, 'pco.npy'))

    while 1:  # close the window to end the program
        pass


if __name__ == '__main__':
    main()
