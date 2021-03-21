import time

import airobot as ar


def print_pose(robot):
    """
    print the robot pose information.

    Args:
        robot (AIRobot): a robot instance of AIRobot.

    """
    pos, quat, rot, euler = robot.arm.get_ee_pose()
    print('End effector pose:')
    print('Position:')
    print(pos)
    print('Euler angles:')
    print(euler)


def main():
    """
    Move the robot end effector to the desired pose.

    The pb=False flag is set because we are using the real robot
    (pb -- pybullet).

    The robot starts at home, and moves to two new cartesian
    poses for the end effector -- one with the orientation fixed,
    and one with a new orientation, both with
    comm_mode use_urscript=False, so the robot will use ROS and MoveIt
    to plan a collision free path.

    The robot then switches comm_mode to use_urscript=True,
    which means it won't use ROS or MoveIt to execute it's commands.

    The robot then repeats the pattern of moving to two new cartesian
    end effector poses using direct end effector pose control with
    urscript commands. These final two movements have been verified
    to be collision free (change the goal poses when use_urscript=True
    at your own risk).
    """
    robot = ar.Robot('ur5e_2f140', pb=False, use_cam=False)
    robot.arm.go_home()
    max_iter = 5

    goal_pos = [0.45, 0.2, 0.3]
    success = False
    k = 0
    while not success and k < max_iter:
        success = robot.arm.set_ee_pose(goal_pos, wait=True)
        print_pose(robot)
        time.sleep(1.0)
        k += 1
    if not success:
        return

    goal_pos = [0.6, -0.15, 0.2]
    goal_ori = [1.57, 0.0, 0.0]
    success = False
    k = 0
    while not success and k < max_iter:
        success = robot.arm.set_ee_pose(goal_pos, goal_ori, wait=True)
        print_pose(robot)
        time.sleep(1.0)
        k += 1
    if not success:
        return

    ar.utils.common.print_red(
        """
        ---SWITCHING COMM MODE TO USE_URSCRIPT=TRUE---\n\n\n
        Robot will execute a joint position command using
        direct position control\n\n\n
        These movements have been verified to be collision free\n\n\n
        ---MODIFY THE GOAL POSITIONS WHEN USING DIRECT POSITION
        CONTROL AT YOUR OWN RISK---
        """
    )
    time.sleep(3.0)

    robot.arm.set_comm_mode(use_urscript=True)

    goal_pos = [0.6, -0.4, 0.2]
    robot.arm.set_ee_pose(goal_pos, wait=True)
    print_pose(robot)


if __name__ == '__main__':
    main()
