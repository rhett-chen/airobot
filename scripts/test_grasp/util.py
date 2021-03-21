import os
import sys
sys.path.append('airobot/')
import airobot.utils.common as ut
import numpy as np
import trimesh
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


def load_pose_6dofgraspnet(pose_path):
    """
    warning: this function only for grasp pose predicted by
     ***6dof-graspnet: <https://github.com/rhett-chen/6dof-graspnet> ***
     pos is bottom point, ori is rotation matrix
    Args:
        pose_path: str, grasp pose and score, .npy file
    Returns: poses, scores
    """

    data = np.load(pose_path).item()
    grasps = data['grasp']
    scores = data['score']
    poses = []
    for pose in grasps:
        center = np.array(pose[:3, 3])   # center is bottom point
        # print(center)
        quat = rot2quat(pose[:3, :3])
        approching_vec = get_approaching_vec(quat)
        poses.append([center, quat, approching_vec])
    return poses, scores


def load_pose_GPNet(pose_path):
    """
    Warning: this function only for grasp pose predicted by ***GPNet***
       pos is center of two contact points, ori is quaternion
    for GPNet repository, the return hasn't contain scores currently
    Args:
        pose_path: str, grasp pose txt file path or npy file path
    Returns: poses, scores
    """

    poses = []
    scores = None
    if pose_path.split('.')[-1] == 'npy':
        data = np.load(pose_path)
        for center, quat in data:
            # center = center + obj_pos
            approching_vec = get_approaching_vec(quat)
            poses.append([center, quat, approching_vec])
    elif pose_path.split('.')[-1] == 'txt':
        with open(pose_path, 'r') as t:
            lines = t.readlines()
            for cand in lines:
                cand = cand.strip()
                center_str = cand.split(']')[0][1:].split()
                center = np.array([float(p) for p in center_str])
                # center += obj_pos
                quat_str = cand.split('[')[-1][:-1].split()
                quat = np.array([float(p) for p in quat_str])

                approaching_vec = get_approaching_vec(quat)
                poses.append([center, quat, approaching_vec])
    else:
        raise NotImplementedError
    return poses, None


def get_approaching_vec(ori):
    """
    Warning: this function assumes when the euler angles formation of orientation is [0, 0, 0], the approaching
            vector is [0, 0, 1]. So it actually calculate a vector rotation
    Args:
        ori: orientation of grasp pose, can be rotation matrix, euler angles or quaternion
    Returns: the approaching vector of orientation
    """
    ori = np.array(ori)
    if ori.size == 3:
        ori = ut.euler2rot(ori)  # [roll, pitch, yaw]
    elif ori.size == 4:
        ori = ut.quat2rot(ori)
    elif ori.shape != (3, 3):
        raise ValueError('Orientation should be rotation matrix, euler angles or quaternion')
    return ori.dot(np.array([0, 0, 1]).T)


def load_single_object(robot, args):
    """
    Warnings: you need to set object type and size
    Args:
        robot:
        args: args.object_pos, args.object_ori

    """
    args.object_pos[2] += args.robot_z_offset
    box_1_id = robot.pb_client.load_geom(
        # 'box',
        # size=0.025,
        'cylinder',   # object type
        size=[0.022, 0.1],  # object size
        mass=0.3,
        base_pos=args.object_pos,
        base_ori=euler2quat(args.object_ori),
        rgba=[1, 0, 0, 1])
    return box_1_id


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


if __name__ == '__main__':
    poses = load_pose_GPNet('data_0/grasp_pose_0.txt', [0, 0, 0])
    poses = poses[:, :2]
    print(type(poses))
    np.save('data_0/grasp_pose_0.npy', poses)
