from ultralytics import YOLO
import cv2
import numpy as np
import time
import traceback
from queue import Empty
import robomaster
from robomaster import robot
from robomaster import camera
from reset_arm import reset_arm


model = YOLO(
    "C:\\Users\\ezyag\\OneDrive\\Desktop\\UMD\\cmsc477\\Homework\\final\\best.pt"
)


# -------------------- STATES --------------------
SEARCH = 0
APPROACH = 1
STOP = 2

# ----------- BOTTOM_Y THRESHOLDS BY CLASS --------
BOTTOM_Y_THRESH = {
    "large_brick": 345,
    "small_brick": 335,
}
BOTTOM_Y_THRESH_DEFAULT = 249
# -------------------------------------------------

# ---------------- YOLO DETECTOR ------------------
class YOLOBlockDetector:
    def __init__(self, model):
        self.model = model

    def find_blocks(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_name = self.model.names[int(box.cls[0])]
            detections.append((x1, y1, x2, y2, conf, cls_name))

        return detections

# returns the length of the bottom of the block
#  what we use to determine how far from and centered on the block the robot is 
def get_block_measurements(detection, img_width):
    x1, y1, x2, y2, conf, _ = detection

    cx = (x1 + x2) / 2.0
    x_error = (cx - img_width / 2) / (img_width / 2)

    bottom_y = y2  # better distance proxy

    return x_error, bottom_y


# draws box around the block
def draw_detections(frame, detections):
    for (x1, y1, x2, y2, conf, cls_name) in detections:
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            cls_name,
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return

# tracks the rotations of the robot (a RoboMaster SDK method)
# unused currently; goal is use this to help robot know where it is in the
# environment (maybe?)
def sub_attitude_info_handler(attitude_info):
    yaw, pitch, roll = attitude_info
    #print("chassis attitude: yaw:{0}, pitch:{1}, roll:{2} ".format(yaw, pitch, roll))


# there were latency issues where the function would stall for a bit but the 
# robot would keep moving so to fix it, i implemented this:
# have the robot move forward for {duration} time then stop
def pulse_drive(ep_chassis, x=0, y=0, z=0, duration=0.05):
    ep_chassis.drive_speed(x=x, y=y, z=z)
    time.sleep(duration)
    ep_chassis.drive_speed(x=0, y=0, z=0)

# this detects and approaches a block and updates state accordingly
# currently: is doesn't get close enough so i use move_closer()
def detect_block_loop(ep_robot, ep_chassis, ep_camera, detector, target_class=None):
    state = SEARCH          # starting state is SEARCH

    # ----------- WHAT YOU TUNE TO ADJUST ALIGNMENT ----------------
    CENTER_THRESH = 0.05        # how centered is centered enough
    bottom_y_thresh = BOTTOM_Y_THRESH.get(target_class, BOTTOM_Y_THRESH_DEFAULT)
    # --------------------------------------------------------------

    while state != STOP: #while state != STOP....
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5) # get a camera frame
        except Empty:
            time.sleep(0.001)
            continue

        detections = detector.find_blocks(img) # retrieve the detections

        if target_class is not None:
            detections = [d for d in detections if d[5] == target_class]

        if len(detections) > 0: # if there are blocks found
            # Pick closest block (largest box)
            detections.sort(key=lambda d: (d[2] - d[0]), reverse=True)
            detection = detections[0]

            # get the x_error and bottom_y vlaues of the detected block
            x_error, bottom_y = get_block_measurements(
                detection, img.shape[1]
            )

            # switch state to APPROACH so robot moves closer
            if state == SEARCH:
                state = APPROACH

            if state == APPROACH: # if state is set to APPROACH
                print(f"x_error: {x_error:.3f}, bottom_y: {bottom_y}") # print alignment information

                # 1. PRIORITY: center first (no forward motion)
                if abs(x_error) > CENTER_THRESH:
                    turn = 5 if x_error > 0 else -5 # tunable: rotate 5 degrees left or right to align
                    ep_chassis.drive_speed(x=0, y=0, z=turn)

                # 2. Once centered, move forward
                elif bottom_y < bottom_y_thresh:
                    pulse_drive(ep_chassis, x=0.15, duration=0.1) #tuneable: speed and duration of time robot moves forward

                # 3. Only stop when centered AND close
                else:
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    state = STOP # switch state back to STOP

        # display state on screen
        draw_detections(img, detections)
        cv2.putText(
            img,
            f"STATE: {state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord("q"):
            break
        
    return

def reset_arm(ep_robot):
    ep_arm = ep_robot.robotic_arm
    gripper = ep_robot.gripper
    print("[reset] Adjusting arm")
    action = ep_arm.moveto(220, -150)
    start = time.time()
    while not action.is_completed:
        if time.time() - start > 10:
            print("[reset] moveto timeout, moving on")
            ep_arm.stop()
            break
        time.sleep(0.05)
    print("[reset] Opening gripper")
    for _ in range(3):
        gripper.open()
        time.sleep(0.3)
    return


def pick_up(ep_robot):
    ep_arm = ep_robot.robotic_arm
    gripper = ep_robot.gripper

    print("[pickup] Raising arm before closing gripper")
    action = ep_arm.moveto(210, 15)
    start = time.time()

    while not action.is_completed:
        if time.time() - start > 5:
            print("[arm] moveto timeout, stopping arm")
            ep_arm.stop()
            return False
        time.sleep(0.05)

    print("[pickup] Closing gripper")
    gripper.close(power=50)
    time.sleep(1)

    print("[pickup] Lifting arm")
    action = ep_arm.moveto(0, 50)
    start = time.time()

    while not action.is_completed:
        if time.time() - start > 5:
            print("[arm] moveto timeout, stopping arm")
            ep_arm.stop()
            return False
        time.sleep(0.05)
    print("Pickup complete.")
    return

# tunable: move closer after yolo approach
# TO DO?: make this function tunable by the caller so you 
# pass in the x and duration you want... similar to pulse_drive()
def move_up(ep_robot):
    ep_chassis = ep_robot.chassis
    print("[move-up] moving up")
    pulse_drive(ep_chassis, x=0.15, duration=1)
    return


def main():
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    detector = YOLOBlockDetector(model)

    try:
        # approach block using YOLO
        reset_arm(ep_robot)
        detect_block_loop(ep_robot, ep_chassis, ep_camera, detector, target_class="small_brick") 
        # move_up(ep_robot) # move closer so you're able to grip
        pick_up(ep_robot)
        
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_camera.stop_video_stream()
        ep_robot.close()
        


if __name__ == "__main__":
    main()
    #explore_main()