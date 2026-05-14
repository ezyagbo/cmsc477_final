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
    pick_up_small,
    reset_arm,
)

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
    print("\n[explore] Exploring — looking for small_brick...")

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
        large_bricks = [d for d in detections if d[5] == "small_brick"]

        if large_bricks:
            best  = max(large_bricks, key=lambda d: (d[2] - d[0]))
            width = best[2] - best[0]
            print(f"[explore] small_brick detected (width={width:.0f}px)")
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
    print(f"\n[explore] Exploring with ±45° sweep — looking for AprilTags: {target_ids}...")

    SWEEP_SPEED = 15    # deg/s
    SWEEP_MAX   = 45.0  # degrees each side

    sweep_angle = 0.0   # accumulated heading offset from straight-ahead
    sweep_dir   = 1     # +1 = sweeping left, -1 = sweeping right
    last_t      = time.time()

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

        tags             = detector.find_tags(frame)
        detected_targets = [t for t in tags if t.tag_id in target_ids]

        if detected_targets:
            target = detected_targets[0]
            print(f"[explore] Target tag {target.tag_id} detected!")
            ep_chassis.drive_speed(x=0, y=0, z=0)
            print("[explore] Tag found — switching to next behavior.")
            return target.tag_id

        state, x_spd, y_spd, z_spd, debug = controller.update(frame)

        now    = time.time()
        dt     = now - last_t
        last_t = now

        if state == "EXPLORE":
            # Accumulate sweep angle and flip direction at ±45°
            sweep_angle += SWEEP_SPEED * sweep_dir * dt
            if sweep_angle >= SWEEP_MAX:
                sweep_angle = SWEEP_MAX
                sweep_dir   = -1
            elif sweep_angle <= -SWEEP_MAX:
                sweep_angle = -SWEEP_MAX
                sweep_dir   = 1
            z_spd = SWEEP_SPEED * sweep_dir
        else:
            # Obstacle avoidance has control — reset sweep so it doesn't fight back
            sweep_angle = 0.0
            sweep_dir   = 1

        ep_chassis.drive_speed(x=x_spd, y=y_spd, z=z_spd, timeout=0.1)

        cv2.imshow("Explore", debug)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            ep_chassis.drive_speed(x=0, y=0, z=0)
            print("[explore] Quit pressed.")
            return None
        
def turn_around(ep_chassis,speed=15):
    """
    Helper function to turn the robot 180.
    """
    ep_chassis.move(x=-.5, y=0, z=0, z_speed=speed).wait_for_completed()
    ep_chassis.move(x=0, y=0, z=180, z_speed=speed).wait_for_completed()
    ep_chassis.drive_speed(x=0, y=0, z=0)

    return

def turn_left(ep_chassis,speed=15):
    """
    Helper function to turn the robot 180.
    """
    ep_chassis.move(x=-.65, y=0, z=0, z_speed=speed).wait_for_completed()
    ep_chassis.move(x=0, y=0, z=150, z_speed=speed).wait_for_completed()
    ep_chassis.drive_speed(x=0, y=0, z=0)

    return

def charge_battery(ep_chassis, ep_camera, controller, tag_detector):
    explore_until_tag(ep_chassis, ep_camera, controller, tag_detector, target_ids=[34, 38])
    detect_tag_loop(ep_chassis, ep_camera, tag_detector, target_id=[34, 38])
    time.sleep(6)

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

        # explore_until_brick(ep_chassis, ep_camera, controller, detector)
        #turn_left(ep_chassis)
        # turn_around(ep_chassis)
        charge_battery(ep_chassis, ep_camera, controller, tag_detector)

          # ── 2. Explore until a small_brick is seen ─────────────────────
        explore_until_brick(ep_chassis, ep_camera, controller, detector)

        # ── 3. Approach and pick up ────────────────────────────────────
        print("\n[pick] Centering on and approaching small_brick...")
        detect_block_loop(
            ep_robot, ep_chassis, ep_camera,
            detector, target_class="small_brick"
        )
        pick_up_small(ep_robot)
        turn_around(ep_chassis) #90 degrees
        explore_until_tag(ep_chassis, ep_camera, controller, tag_detector, target_ids=[41, 11])
        detect_tag_loop(ep_chassis, ep_camera, tag_detector, target_id=[41, 11])
        reset_arm(ep_robot)

        # # --- ROUND 2 -----------
        print("\n[system] Round 1 complete. Press Enter or 'g' to start Round 2...")
        while True:
            key = input().strip().lower()
            if key in ("", "g"):
                break

        turn_around(ep_chassis)

        charge_battery(ep_chassis, ep_camera, controller, tag_detector)

          # ── 2. Explore until a small_brick is seen ─────────────────────
        explore_until_brick(ep_chassis, ep_camera, controller, detector)

        # ── 3. Approach and pick up ────────────────────────────────────
        print("\n[pick] Centering on and approaching small_brick...")
        detect_block_loop(
            ep_robot, ep_chassis, ep_camera,
            detector, target_class="small_brick"
        )
        pick_up_small(ep_robot)
        turn_left(ep_chassis) #90 degrees 
        explore_until_tag(ep_chassis, ep_camera, controller, tag_detector, target_ids=[41, 11])
        detect_tag_loop(ep_chassis, ep_camera, tag_detector, target_id=[41, 11])
        reset_arm(ep_robot)

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
