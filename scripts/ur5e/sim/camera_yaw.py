import matplotlib.pyplot as plt
import numpy as np

from airobot import Robot


def main():
    """
    This function demonstrates how the yaw angle (
     the yaw angle that is defined in robot.setup_camera) changes
    the camera view.
    """
    robot = Robot('ur5e')
    focus_pt = [0, 0, 1]  # ([x, y, z])
    robot.arm.go_home()
    img = np.random.rand(480, 640)
    image = plt.imshow(img, interpolation='none',
                       animated=True, label="cam")
    ax = plt.gca()
    while True:
        for yaw in range(0, 360, 10):
            robot.cam.setup_camera(focus_pt=focus_pt,
                                   dist=3,
                                   yaw=yaw,
                                   pitch=0,
                                   roll=0)

            rgb, depth = robot.cam.get_images(get_rgb=True,
                                              get_depth=True)
            image.set_data(rgb)
            ax.plot([0])
            plt.pause(0.01)


if __name__ == '__main__':
    main()
