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
from approach_tag import (
    AprilTagDetector,
    detect_tag_loop,
)

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
        
        print("[debug] calling controller.update...") # Add this
        state, x_spd, y_spd, z_spd, debug = controller.update(frame)
        print(f"[debug] speeds: {x_spd}, {z_spd}")    # Add this
        
        ep_chassis.drive_speed(x=x_spd, y=y_spd, z=z_spd, timeout=0.1)

        if debug is not None:
            cv2.imshow("Explore", debug)
        
        # Ensure waitKey is always called
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return
        
def explore_until_tag(ep_chassis, ep_camera, controller, detector, target_ids):
    """
    Run the ObstacleController explore loop and return as soon as an 
    AprilTag with one of the target_ids is detected.
    
    :param target_ids: List or tuple of IDs to look for (e.g., [19, 45])
    """
    print(f"\n[explore] Exploring — looking for AprilTags: {target_ids}...")

    while True:
        try:
            # Get the latest frame from the camera
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Empty:
            continue
        except Exception as e:
            print(f"[WARN] Camera error: {e}")
            continue

        if frame is None:
            continue

        # Use the detector to find AprilTags in the current frame
        # (Assuming detector.find_tags returns objects with an .id property)
        tags = detector.find_tags(frame)
        
        
        # Filter for tags that match our target list
        detected_targets = [t for t in tags if t.tag_id in target_ids]

        if detected_targets:
            # Pick the "best" tag based on size (center distance or area)
            # Here we just take the first one found or you could use area/width
            target = detected_targets[0]
            print(f"[explore] Target tag {target.tag_id} detected!")
            
            # Stop the robot immediately
            ep_chassis.drive_speed(x=0, y=0, z=0)
            print("[explore] Tag found — switching to next behavior.")
            return target.tag_id  # Return the ID found so you know which one triggered it

        # If no tag is seen, continue the exploration movement
        state, x_spd, y_spd, z_spd, debug = controller.update(frame)
        ep_chassis.drive_speed(x=x_spd, y=y_spd, z=z_spd, timeout=0.1)

        # Visual feedback
        cv2.imshow("Explore", debug)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ep_chassis.drive_speed(x=0, y=0, z=0)
            print("[explore] Quit pressed.")
            return None
        
def turn_around(ep_chassis,speed=15):
    """
    Helper function to turn the robot 180.
    """
    ep_chassis.move(x=0, y=0, z=180, z_speed=speed).wait_for_completed()
    ep_chassis.drive_speed(x=0, y=0, z=0)

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
    tag_detector = AprilTagDetector()

    print("[init] Starting camera stream...")
    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
    time.sleep(0.5)

    try:
        #── 1. Prep arm ────────────────────────────────────────────────
        reset_arm(ep_robot)

        # # ── 2. Explore until a large_brick is seen ─────────────────────
        # explore_until_brick(ep_chassis, ep_camera, controller, detector)

        # # ── 3. Approach and pick up ────────────────────────────────────
        # print("\n[pick] Centering on and approaching large_brick...")
        # detect_block_loop(
        #     ep_robot, ep_chassis, ep_camera,
        #     detector, target_class="large_brick"
        # )
        # pick_up(ep_robot)
        # turn_around(ep_chassis)
        explore_until_tag(ep_chassis, ep_camera, controller, tag_detector, target_ids=[21])
        detect_tag_loop(ep_chassis, ep_camera, tag_detector, target_id=21)
        # reset_arm(ep_robot)

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
