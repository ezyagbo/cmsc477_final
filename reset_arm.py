import time
import cv2
import numpy as np
import robomaster
from robomaster import robot, camera
from ultralytics import YOLO
from enum import Enum

def reset_arm(ep_robot):
    ep_arm = ep_robot.robotic_arm
    gripper = ep_robot.gripper
    print("[reset] Adjusting arm")
    ep_arm.moveto(220, -150).wait_for_completed()   
    print("[reset] Opening gripper")
    gripper.open()
    return




if __name__ == '__main__':
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    try:
        reset_arm(ep_robot)
    finally:
        ep_robot.close()