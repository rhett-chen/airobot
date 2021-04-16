import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import airobot.utils.common as ut
import numpy as np
import trimesh
import mayavi.mlab as mlab
from airobot.utils.common import euler2quat, rot2quat


def pos_offset_along_ori(orientation, offset_dis):
    """
    Args:
        orientation: approaching vector, 3-d list
        offset_dis: the offset distance along the approaching vector, positive means along, negative means opposite
    Returns: 3-d numpy array, the original coordinate plus this return value, you can get the translated coordinate
            along approaching vector
    """
    denominator = np.sqrt(orientation[0] ** 2 + orientation[1] ** 2 + orientation[2] ** 2)
    offset_z = orientation[2] / denominator * offset_dis
    offset_y = orientation[1] / denominator * offset_dis
    offset_x = orientation[0] / denominator * offset_dis
    return np.array([offset_x, offset_y, offset_z])


def load_pose_graspnet_baseline(pose_path):
    """
        warning: this function only for grasp pose predicted by
         ***graspnet-baseline: <https://github.com/rhett-chen/graspnet-baseline> ***
         pos is bottom point, ori is rotation matrix
         grasp pose format is recorded in https://graspnetapi.readthedocs.io/en/latest/grasp_format.html
        Args:
            pose_path: str, grasp pose and score, .npy file
        Returns: poses(quaternion(for x-z plane), center(center of two contact points), approaching vec), scores
        """
    data = np.load(pose_path)
    sorted(data, key=lambda x: x[0], reverse=True)
    scores = data[:, 0]
    rotation_matrix = data[:, 4:13].reshape((-1, 3, 3))
    centers = data[:, 13:16]
    poses = []
    for index, rot in enumerate(rotation_matrix):
        # quat = ut.rot2quat(rot)
        approaching_vec = get_approaching_vec(rot, up_vector=[1, 0, 0])
        poses.append([centers[index], rot, approaching_vec])

    # see why we do the rotation and offset in util.py -> get_control_points()
    for pose in poses:
        rot = pose[1]
        te_1 = ut.euler2rot(np.array([0, np.pi / 2, 0]))
        te_2 = ut.euler2rot(np.array([np.pi / 2, 0, 0]))
        final_rot = np.dot(np.dot(rot, te_2), te_1)
        pose[1] = ut.rot2quat(final_rot)

    offset_along_ori = 0.02
    for center, _, vec in poses:
        center += pos_offset_along_ori(vec, offset_along_ori)
    return poses, scores


def load_pose_6dofgraspnet(pose_path):
    """
    warning: this function only for grasp pose predicted by
     ***6dof-graspnet: <https://github.com/rhett-chen/6dof-graspnet> ***
     pos is bottom point, ori is rotation matrix
    Args:
        pose_path: str, grasp pose and score, .npy file
    Returns: poses(quaternion(for x-z plane), center(center of two contact points), approaching vec), scores
    """

    data = np.load(pose_path).item()
    grasps = data['grasp']
    scores = data['score']
    poses = []
    for pose in grasps:
        center = np.array(pose[:3, 3])
        quat = rot2quat(pose[:3, :3])
        approching_vec = get_approaching_vec(quat)
        poses.append([center, quat, approching_vec])

    # center is bottom point, change it to center of two contact points, offset=0.10527 is the distance from bottom
    # point to center of two contact points, you can see comments of util.py -> get_grasp_points() for details
    offset_along_ori = 0.10527
    for center, _, vec in poses:
        center += pos_offset_along_ori(vec, offset_along_ori)
    return poses, scores


def load_pose_GPNet(pose_path):
    """
    Warning: this function only for grasp pose predicted by ***GPNet***
       pos is center of two contact points, ori is quaternion
    for GPNet repository, the return hasn't contain scores currently
    Args:
        pose_path: str, grasp pose txt file path or npy file path
    Returns:  poses(quaternion, center(center of two contact points), approaching vec), scores
    """

    poses = []
    scores = None
    data = np.load(pose_path)
    for center, quat in data:
        # center = center + obj_pos
        approching_vec = get_approaching_vec(quat)
        poses.append([center, quat, approching_vec])
    return poses, None


def get_approaching_vec(rot, up_vector=None):
    """
    Warning: this function assumes when the euler angles formation of rotation is [0, 0, 0], the approaching
            vector is up_vector. So it actually calculate a vector rotation
    Args:
        up_vector: approaching vector when rotation is [0, 0, 0] in euler angle
        rot: rotation of grasp pose, can be rotation matrix, euler angles or quaternion
    Returns: the approaching vector of orientation
    """
    if up_vector is None:
        up_vector = [0, 0, 1]
    rot = np.array(rot)
    up_vector = np.array(up_vector)
    if rot.size == 3:
        rot = ut.euler2rot(rot)  # [roll, pitch, yaw]
    elif rot.size == 4:
        rot = ut.quat2rot(rot)
    elif rot.shape != (3, 3):
        raise ValueError('Orientation should be rotation matrix, euler angles or quaternion')
    return rot.dot(up_vector.T)


def load_single_object(robot, args):
    """
    Warnings: you need to set object type and size
    Args:
        robot:
        args: args.object_pos, args.object_ori

    """
    args.object_pos[2] += args.robot_z_offset
    obj_id = robot.pb_client.load_geom(
        'box',
        size=0.025,
        # 'cylinder',   # object type
        # size=[0.022, 0.1],  # object size
        mass=0.3,
        base_pos=args.object_pos,
        base_ori=euler2quat(args.object_ori),
        rgba=[1, 0, 0, 1])
    return obj_id


def load_multi_object(robot, args):
    """
    Warnings: you need to set object type and size
    Args:
        robot:
        args: robot_z_offset
    """
    z_offset = args.robot_z_offset
    obj_1_id = robot.pb_client.load_geom(
        'box',
        size=0.025,
        # 'cylinder',   # object type
        # size=[0.022, 0.1],  # object size
        mass=0.3,
        base_pos=[0.3, 0, 0.1 + z_offset],
        base_ori=euler2quat([0, 0, 0.3]),
        rgba=[1, 0, 0, 1])

    obj_2_id = robot.pb_client.load_geom(
        'cylinder',   # object type
        size=[0.023, 0.05],  # object size
        mass=0.3,
        base_pos=[0.25, 0.045, 0.1 + z_offset],
        base_ori=euler2quat([0, 0, 0.13]),
        rgba=[1, 0, 0, 1])

    obj_3_id = robot.pb_client.load_geom(
        'capsule',
        size=[0.02, 0.037],
        mass=0.3,
        base_pos=[0.35, -0.07, 0.05 + z_offset],
        base_ori=euler2quat([0, 0, 0.63]),
        rgba=[1, 0, 0, 1])

    obj_4_id = robot.pb_client.load_geom(
        'box',
        size=[0.025, 0.05, 0.02],
        mass=0.3,
        base_pos=[0.36, 0, 0.1 + z_offset],
        base_ori=euler2quat([0, 0, 1.33]),
        rgba=[1, 0, 0, 1])
    return obj_1_id, obj_2_id, obj_3_id, obj_4_id


def get_grasp_points():
    """
       grasp points cloud contain 7 points, during visualization, the 7 points will be connected in sequence to form a
    shape of gripper. Originally, there are two gripper versions: panda and customized, difference between them is shown
    in the following comments. Now delete panda version, only use customized version.

        pc_gripper_version: customized | panda
                                 |  ^  |   <- top is two contact points, ^ is center of two contact points
            gripper figure:      |_____|   <- line
                                    |
                                    !     <- bottom is bottom point
          customized_grasp_points.npy: the center of two contact points is at [0, 0, 0],
                two contact points are [0.037, 0, 0] [-0.037, 0, 0]
                line: [0.037, 0, -0.046], [-0.037, 0, -0.046]
                bottom point: [0, 0, -0.105]
          panda.npy: contact point: [0.0526874, 0, 0.10527] [-0.0526874, 0, 0.10527]
                the center of two contact points is at [0, 0, 0.10527],
                line: [0.0526874, 0, 0.059] [-0.0526874, 0, 0.059]
                bottom point: [0, 0, 0]
         For the above two files, the two-fingers plane is x-z plane.
         For customized version, center (two contact points center) is [0, 0, 0], bottom point is [0, 0, -0.105]
         For panda version, bottom point is [0, 0, 0], center of two contact points is [0, 0, 0.10527];

            1. grasp poses generate by GPNet: pos is center of two contact points; rotation is for gripper in x-z plane
            2. grasp poses generated by 6dof-graspnet: pos is bottom point of panda, offset is 0.10527, from bottom
         point to the center of two contacted points along approaching vector, rotation is for gripper in x-z plane,
         so in load_pose_6dofgraspnet(), we add offset to the original poses.
            3. grasp poses generated by graspnet-baseline: pos is 0.02 away from center of two contact points in the
         opposite direction of the approaching vector. rotation is for gripper in x-y plane, towards x-axis, so in
         load_pose_graspnet_baseline(), we add rotation and offset to the original poses.
    Returns: grasp points shape [7, 3]
    """
    # original panda's control_points
    # control_points = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    #                            [0.00000000e+00, 0.00000000e+00, 5.90000004e-02],
    #                            [5.26874326e-02, -5.99553132e-05, 5.90000004e-02],
    #                            [5.26874326e-02, -5.99553132e-05, 1.05273142e-01],
    #                            [5.26874326e-02, -5.99553132e-05, 5.90000004e-02],
    #                            [-5.26874326e-02, 5.99553132e-05, 5.90000004e-02],
    #                            [-5.26874326e-02, 5.99553132e-05, 1.05273142e-01]])

    # customized control points
    control_points = np.array([[0., 0., -0.105], [0., 0., -0.046], [0.037, 0, -0.046], [0.037, 0, 0.],
                               [0.037, 0, -0.046], [-0.037, 0, -0.046], [-0.037, 0, 0.]])
    return control_points


def get_color_plasma(x):
    return tuple([float(1 - x), float(x), float(0)])


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


class Object(object):
    """Represents a graspable object."""

    def __init__(self, filename):
        """Constructor.
        :param filename: Mesh to load
        :param scale: Scaling factor
        """
        self.mesh = trimesh.load(filename)
        self.scale = 1.0

        # print(filename)
        self.filename = filename
        if isinstance(self.mesh, list):
            # this is fixed in a newer trimesh version:
            # https://github.com/mikedh/trimesh/issues/69
            print("Warning: Will do a concatenation")
            self.mesh = trimesh.util.concatenate(self.mesh)

        self.collision_manager = trimesh.collision.CollisionManager()
        self.collision_manager.add_object('object', self.mesh)

    def rescale(self, scale=1.0):
        """Set scale of object mesh.
        :param scale
        """
        self.scale = scale
        self.mesh.apply_scale(self.scale)

    def resize(self, size=1.0):
        """Set longest of all three lengths in Cartesian space.
        :param size
        """
        self.scale = size / np.max(self.mesh.extents)
        self.mesh.apply_scale(self.scale)

    def in_collision_with(self, mesh, transform):
        """Check whether the object is in collision with the provided mesh.
        :param mesh:
        :param transform:
        :return: boolean value
        """
        return self.collision_manager.in_collision_single(mesh, transform=transform)


def draw_scene_mayavi(args,
               pc,
               grasps=None,
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               show_gripper_mesh=False,
               grasps_selection=None,
               visualize_diverse_grasps=False,
               min_seperation_distance=0.03,
               pc_color=None,
               plasma_coloring=False,
               target_cps=None):
    """
    changed based on https://github.com/jsll/pytorch_6dof-graspnet, you can find original version here
    Draws the 3D scene for the object and the scene.
    Args:
      args: util.make_parser(), parameters needed for data acquisition, grasp, create robot, grasp visualization, etc
      gripper_color: gripper color in rgb
      pc: point cloud of the object
      grasps: list of numpy array indicating the transformation of the grasps: [position, rotation, approaching vector]
      grasp_scores: grasps will be colored based on the scores. If left empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the
            top score grasp to visualize and then choose grasps that are not within
            min_seperation_distance distance of any of the previously selected
            grasps. Only set it to True to declutter the grasps for better visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizing the pc.
    """
    if grasps is None:
        grasps = []
    max_grasps = 45
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:
        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0, high=len(grasps), size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc[:, 2],  # coloring point according to height
                          colormap='plasma')
        else:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          color=(0.1, 0.1, 1),
                          scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc_color[:, 0],
                          colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

    grasp_pc = get_grasp_points()

    # a = ut.euler2rot(np.array([0, np.pi / 2, 0]))
    # grasp_pc = np.dot(a, grasp_pc.T)
    # a = ut.euler2rot(np.array([np.pi / 2, 0, 0]))
    # grasp_pc = np.dot(a, grasp_pc).T
    # print(grasp_pc)

    if grasp_scores is None:
        indexes = range(len(grasps))
    else:
        indexes = np.argsort(-np.asarray(grasp_scores))
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    print('draw scene ', len(grasps))

    selected_grasps_so_far = []
    removed = 0

    for ii in range(len(grasps)):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] is False:
                continue

        g = grasps[i]
        is_diverse = True
        for prevg in selected_grasps_so_far:
            distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])
            if distance < min_seperation_distance:
                is_diverse = False
                break

        if visualize_diverse_grasps:
            if not is_diverse:
                removed += 1
                continue
            else:
                if grasp_scores is not None:
                    print('selected', i, grasp_scores[i], min_score, max_score)
                else:
                    print('selected', i)
                selected_grasps_so_far.append(g)

        if isinstance(gripper_color, list):
            pass
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] - min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                gripper_color = grasp_color[ii]
            else:
                gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                gripper_color = (0.0, 1.0, 0.0)

        if show_gripper_mesh:
            if args.grasp_points_file[0] != 'p':
                raise NotImplementedError
            gripper_mesh = Object(
                'panda_gripper.obj').mesh
            gripper_mesh.apply_transform(g)
            mlab.triangular_mesh(
                gripper_mesh.vertices[:, 0],
                gripper_mesh.vertices[:, 1],
                gripper_mesh.vertices[:, 2],
                gripper_mesh.faces,
                color=gripper_color,
                opacity=1 if visualize_diverse_grasps else 0.5)
        else:
            pts = np.matmul(grasp_pc, ut.to_rot_mat(g[1]).T)  # rotation
            pts += np.expand_dims(g[0], 0)  # translation
            if isinstance(gripper_color, list):
                mlab.plot3d(pts[:, 0],
                            pts[:, 1],
                            pts[:, 2],
                            color=gripper_color[i],
                            tube_radius=0.003,
                            opacity=1)
            else:
                tube_radius = 0.001
                mlab.plot3d(pts[:, 0],
                            pts[:, 1],
                            pts[:, 2],
                            color=gripper_color,
                            tube_radius=tube_radius,
                            opacity=1)
                if target_cps is not None:
                    mlab.points3d(target_cps[ii, :, 0],
                                  target_cps[ii, :, 1],
                                  target_cps[ii, :, 2],
                                  color=(1.0, 0.0, 0),
                                  scale_factor=0.01)

    print('removed {} similar grasps'.format(removed))


if __name__ == '__main__':
    poses = load_pose_GPNet('data_0/grasp_pose_0.txt', [0, 0, 0])
    poses = poses[:, :2]
    print(type(poses))
    np.save('data_0/grasp_pose_0.npy', poses)
