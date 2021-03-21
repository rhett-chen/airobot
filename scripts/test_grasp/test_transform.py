import airobot.utils.common as co
import numpy as np
import mayavi.mlab as mlab
import cv2

# origin = np.array([1.2, 2, 1])
# euler = [0.9, 0.6, -np.pi / 2]
# rot_mat = co.to_rot_mat(euler)
# print(rot_mat)
# ans = np.dot(rot_mat, origin.T)  # Rotate the vector in the world coordinate system
# print(ans)
# print(np.dot(origin, rot_mat.T))  # origin * rot_mat.T == rot_mat * orijin.T
# print(np.dot(rot_mat.T, ans.T))  # answer is origin vector

# euler = [0, 0, 0]
# rot = co.to_rot_mat(euler)
# print(rot)
#
# panda_pc = np.array([[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
#                      [0.0000000e+00, 0.0000000e+00, 5.9000000e-02],
#                      [5.2687433e-02, -5.9955313e-05, 5.9000000e-02],
#                      [5.2687433e-02, -5.9955313e-05, 1.0527314e-01],
#                      [5.2687433e-02, -5.9955313e-05, 5.9000000e-02],
#                      [-5.2687433e-02, 5.9955313e-05, 5.9000000e-02],
#                      [-5.2687433e-02, 5.9955313e-05, 1.0527314e-01]])
# np.save('panda_grasp_points.npy', panda_pc)

# my_pc = np.array([[0., 0., -0.105], [0., 0., -0.046], [0.037, 0, -0.046],
#                   [0.037, 0, 0.], [0.037, 0, -0.046], [-0.037, 0, -0.046], [-0.037, 0, 0.]])
# np.save('customized_grasp_points.npy', my_pc)
# a = ['pp': lambda _ : print('vsdv')]


# def dispatch_dict(op, x, y):
#     return {
#         'aa': print(x+y)
#     }.get(op)
#
#
# dispatch_dict('aa', 2, 3)
# a = [0, 0, 0]
# print(co.to_rot_mat(a))

# data = np.load('cylinder.npy', allow_pickle=True, encoding="latin1").item()
# pc = data['smoothed_object_pc']

# pc = np.load('train_pc.npy')
# print(pc.shape)
# mlab.points3d(pc[:, 0],
#               pc[:, 1],
#               pc[:, 2],
#               color=(0.1, 0.1, 1),
#               scale_factor=0.01)
# mlab.show()

# image = data['image']
# print(image.shape)
# depth = data['depth']
# print(depth.shape)
# cv2.imshow('v', image)
# cv2.waitKey(0)
