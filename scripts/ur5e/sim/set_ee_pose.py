import time

from airobot import Robot
from airobot import log_info


def main():
    """
    Move the robot end effector to the desired pose.
    """
    robot = Robot('ur5e')
    robot.arm.go_home()
    tgt_pos = [0.45, 0.2, 1.3]
    robot.arm.set_ee_pose(tgt_pos)

    # sleep statement is not necessary
    time.sleep(1)
    tgt_pos = [0.6, -0.15, 1.2]
    tgt_euler = [1.57, 0.0, 0.0]
    robot.arm.set_ee_pose(tgt_pos, tgt_euler)
    time.sleep(1)
    pos, quat, rot, euler = robot.arm.get_ee_pose()
    log_info('End effector pose:')
    log_info('Position:')
    log_info(pos)
    log_info('Euler angles:')
    log_info(euler)


if __name__ == '__main__':
    main()
