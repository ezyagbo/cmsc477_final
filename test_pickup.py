"""
test_pickup.py

Full pipeline:
  1. Explore with obstacle avoidance until a large_brick is seen.
  2. Approach and pick up the brick.
  3. Spin to find the dropoff AprilTag, approach it, and drop the brick.
"""

import time
import cv2
import robomaster
from robomaster import robot, camera
from queue import Empty
from ultralytics import YOLO

from explore import ObstacleController
from approach_and_pick_block import (
    YOLOBlockDetector,
    detect_block_loop,
    pick_up,
    reset_arm,
)
from tag_delivery import deliver_block, LARGE_BRICK_DROP_TAG

# ============================================================
# Config
# ============================================================
ROBOT_IP   = "192.168.50.121"
MODEL_PATH = "C:\\Users\\ezyag\\OneDrive\\Desktop\\UMD\\cmsc477\\Homework\\final\\best.pt"

# Min bounding-box width (px) before the brick is treated as "seen".
BRICK_SEEN_MIN_WIDTH = 60


# ============================================================
# Explore phase
# ============================================================

def explore_until_brick(ep_chassis, ep_camera, controller, detector):
    """
    Run the ObstacleController explore loop and return as soon as a
    large_brick wide enough to be close is detected.
    """
    print("\n[explore] Exploring — looking for large_brick...")

    while True:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        except Exception as e:
            print(f"[WARN] Camera error: {e}")
            continue

        if frame is None:
            continue

        detections   = detector.find_blocks(frame)
        large_bricks = [d for d in detections if d[5] == "large_brick"]

        if large_bricks:
            best  = max(large_bricks, key=lambda d: (d[2] - d[0]))
            width = best[2] - best[0]
            print(f"[explore] large_brick detected (width={width:.0f}px)")
            if width >= BRICK_SEEN_MIN_WIDTH:
                ep_chassis.drive_speed(x=0, y=0, z=0)
                print("[explore] Brick close enough — switching to approach.")
                return

        state, x_spd, y_spd, z_spd, debug = controller.update(frame)
        ep_chassis.drive_speed(x=x_spd, y=y_spd, z=z_spd, timeout=0.1)

        cv2.imshow("Explore", debug)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ep_chassis.drive_speed(x=0, y=0, z=0)
            print("[explore] Quit pressed.")
            return


# ============================================================
# Main
# ============================================================

def main():
    robomaster.config.ROBOT_IP_STR = ROBOT_IP

    model = YOLO(MODEL_PATH)

    ep_robot  = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_chassis = ep_robot.chassis
    ep_camera  = ep_robot.camera

    controller = ObstacleController(yolo_model=model)
    detector   = YOLOBlockDetector(model)

    print("[init] Starting camera stream...")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    time.sleep(0.5)

    try:
        # ── 1. Prep arm ────────────────────────────────────────────────
        reset_arm(ep_robot)

        # ── 2. Explore until a large_brick is seen ─────────────────────
        explore_until_brick(ep_chassis, ep_camera, controller, detector)

        # ── 3. Approach and pick up ────────────────────────────────────
        print("\n[pick] Centering on and approaching large_brick...")
        detect_block_loop(
            ep_robot, ep_chassis, ep_camera,
            detector, target_class="large_brick"
        )
        pick_up(ep_robot)

        # ── 4. Deliver to dropoff tag ──────────────────────────────────
        

    except KeyboardInterrupt:
        print("\n[system] Interrupted.")

    finally:
        ep_chassis.drive_speed(x=0, y=0, z=0)
        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass
        ep_robot.close()
        cv2.destroyAllWindows()
        print("[system] Robot disconnected cleanly.")


if __name__ == "__main__":
    main()
