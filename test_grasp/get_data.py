import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from util import load_multi_object, load_single_object, parse_robot_type
import airobot as ar
import cv2
import time
import pybullet as p
import numpy as np
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
    robot_type = parse_robot_type(args.robot_arm)
    robot = ar.Robot(robot_type)
    robot.arm.go_home()
    robot.pb_client.load_urdf('plane.urdf')

    if args.multi_obj:
        _, poses = load_multi_object(robot)
    else:
        load_single_object(robot, args)
    time.sleep(1.3)

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
            'cam_height': args.cam_height, 'cam_width': args.cam_width}
    if args.multi_obj:
        info['object'] = poses
    else:
        info['object'] = [args.object_pos, args.object_ori]

    np.save(os.path.join(save_path, 'info.npy'), info)

    with open(os.path.join(save_path, 'info.txt'), 'w') as ci:
        if args.multi_obj:
            for i, pose in enumerate(poses):
                ci.write('object_{} position:\n'.format(i))
                ci.write(str(pose[0]))
                ci.write('\nobject_{} rotation:\n'.format(i))
                ci.write(str(pose[1]))
        else:
            ci.write('object position: \n')
            ci.write(str(args.object_pos) + '\n')
            ci.write('object rotation: \n')
            ci.write(str(args.object_ori) + '\n')
        ci.write('\ncamera position: \n')
        ci.write(str(args.cam_pos) + '\n')
        ci.write('camera focus point: \n')
        ci.write(str(args.cam_focus_pt) + '\n')
        ci.write('camera resolution: \n')
        ci.write(str(args.cam_height) + ' * ' + str(args.cam_width) + '\n')
        ci.write('camera view matrix: \n')
        ci.write(str(robot.cam.view_matrix) + '\n')
        ci.write('camera intrinsic matrix: ' + '\n')
        ci.write(str(robot.cam.cam_int_mat) + '\n')
        ci.write('camera external matrix: ' + '\n')
        ci.write(str(robot.cam.cam_ext_mat))
        ci.flush()

    # save rgb. depth, segmentation, point cloud
    img, depth, seg = robot.cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
    print('image shape: {}, depth shape: {}'.format(img.shape, depth.shape))
    cv2.imwrite(os.path.join(save_path, 'depth.jpg'),
                encode_depth_to_image(depth))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    cv2.imwrite(os.path.join(save_path, 'rgb.jpg'), img)
    ar.log_info(f'Depth image min:{depth.min()} m, max: {depth.max()} m.')
    np.save(os.path.join(save_path, 'depth.npy'), depth)
    np.save(os.path.join(save_path, 'seg.npy'), seg)  # segmentation
    # get point cloud data in the world frame
    pts = robot.cam.depth_to_point_cloud(depth, in_world=True)
    ar.log_info('point cloud shape: {}'.format(pts.shape))
    np.save(os.path.join(save_path, 'pc.npy'), pts)
    print('saved %s successfully' % os.path.join(save_path, 'pc.npy'))

    input()


if __name__ == '__main__':
    main()
