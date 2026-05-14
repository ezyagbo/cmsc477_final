import pupil_apriltags
from pupil_apriltags import Detector
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
from approach_and_pick_block import main, pulse_drive, SEARCH, APPROACH, STOP


# ============================================================
# AprilTag settings
# ============================================================
MARKER_SIZE_M = 0.153

K = np.array([
    [314, 0, 320],
    [0, 314, 180],
    [0, 0, 1]
], dtype=np.float32)

class AprilTagDetector:
    def __init__(self):
        # windows users may need to specify search paths for the library
        self.camera_params = [
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2]
        ]
        
        self.detector = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=1
        )

    def find_tags(self, frame):
        # Check the number of channels (shape will be (h, w, 3) for BGR or (h, w) for Gray)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # It's a BGR image, convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # It's already grayscale
            gray = frame

        results = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=MARKER_SIZE_M
        )

        # Each 'tag' object has .tag_id and .center (x, y)
        return results
    
    # ----------- TAG THRESHOLDS ---------------------
TAG_Y_THRESH = 165    # Adjust based on how close you want to get
CENTER_THRESH = 0.06  # Deadzone for centering
# ------------------------------------------------

def get_tag_measurements(tag, img_width):
    """
    Returns x_error (normalized -1 to 1) and y_position.
    """
    tx, ty = tag.center
    x_error = (tx - img_width / 2) / (img_width / 2)
    return x_error, ty

def detect_tag_loop(ep_chassis, ep_camera, tag_detector, target_id):
    """
    Centers and approaches a specific AprilTag ID or any ID from a list.
    """
    target_ids = set(target_id) if hasattr(target_id, '__iter__') else {target_id}
    state = SEARCH
    print(f"\n[tag-loop] Seeking Tag ID(s): {target_ids}")

    while state != STOP:
        try:
            img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue

        if img is None:
            continue

        # Detect tags
        tags = tag_detector.find_tags(img)

        # Filter for our specific ID
        target_tags = [t for t in tags if t.tag_id in target_ids]

        if target_tags:
            # If multiple are seen, pick the one lowest in the frame (closest)
            target = max(target_tags, key=lambda t: t.center[1])
            x_error, tag_y = get_tag_measurements(target, img.shape[1])

            if state == SEARCH:
                state = APPROACH

            if state == APPROACH:
                # 1. Centering Priority
                if abs(x_error) > CENTER_THRESH:
                    turn_speed = 5 if x_error > 0 else -5
                    ep_chassis.drive_speed(x=0, y=0, z=turn_speed)

                # 2. Forward Movement
                elif tag_y < TAG_Y_THRESH:
                    print(f"[tag-loop] Centered. Moving closer... (y={tag_y:.0f})")
                    pulse_drive(ep_chassis, x=0.15, duration=0.15)

                # 3. Target Reached
                else:
                    ep_chassis.drive_speed(x=0, y=0, z=0)
                    print(f"[tag-loop] Reached Tag {list(target_ids)}!")
                    state = STOP

        # --- Visual Debugging ---
        for tag in tags:
            cx, cy = int(tag.center[0]), int(tag.center[1])
            color = (0, 255, 0) if tag.tag_id in target_ids else (0, 0, 255)
            cv2.circle(img, (cx, cy), 8, color, -1)
            cv2.putText(img, f"ID: {tag.tag_id}", (cx - 20, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(img, f"STATE: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("AprilTag Tracking", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return True


def main():
    robomaster.config.ROBOT_IP_STR = "192.168.50.121"

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

    detector = AprilTagDetector()

    try:
        # approach block using YOLO
        reset_arm(ep_robot)
        detect_tag_loop(ep_chassis, ep_camera, detector, target_id=34) 
        
    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        ep_camera.stop_video_stream()
        ep_robot.close()
        


if __name__ == "__main__":
   main()
