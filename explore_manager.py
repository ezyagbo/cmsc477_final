import time
import cv2
import robomaster
from robomaster import robot, camera
from queue import Empty
from ultralytics import YOLO

# This imports your partner's movement/avoidance controller.
# Make sure explore.py is in the same folder.
from explore import ObstacleController

# This imports your mapping system.
from mapper import Mapper

#This imports odometry test code.
#from odometry_test_v2 import return_to_start

model = YOLO(
    "C:\\Users\\ezyag\\OneDrive\\Desktop\\UMD\\cmsc477\\Homework\\final\\best.pt"
)

# ============================================================
# Robot IP
# ============================================================
ROBOT_IP = "192.168.50.121"


# ============================================================
# Odometry globals
# ============================================================
current_x = 0.0
current_y = 0.0
current_yaw = 0.0
last_odom_time = 0.0


def position_callback(position_info):
    """
    Called by RoboMaster SDK when odometry data arrives.

    position_info = (x, y, yaw)
    x and y are in meters.
    yaw is in degrees.
    """
    global current_x, current_y, current_yaw, last_odom_time

    current_x, current_y, current_yaw = position_info
    last_odom_time = time.time()


def draw_main_debug(frame, state, tag_events, mapper):
    """
    Adds mapping info on top of the normal camera frame.

    This is separate from explore.py's debug frame.
    """

    out = frame.copy()

    text_lines = [
        f"Explore State: {state}",
        f"Odom: x={current_x:.2f}, y={current_y:.2f}, yaw={current_yaw:.1f}",
        f"Mapped Tags: {len(mapper.mapped_tags)}",
        f"Mapped Obstacles: {len(mapper.obstacles)}",
        f"Tags Seen This Frame: {len(tag_events)}",
        "Press m = print map | s = save map | q = quit"
    ]

    for i, text in enumerate(text_lines):
        cv2.putText(
            out,
            text,
            (10, 30 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return out


def main():
    robomaster.config.ROBOT_IP_STR = ROBOT_IP

    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")

    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera

    # Movement brain from explore.py — pass YOLO model for box avoidance
    explorer = ObstacleController(yolo_model=model)

    # Memory brain from mapper.py
    mapper = Mapper()

    print("[init] Subscribing to odometry...")
    ep_chassis.sub_position(
        cs=1,
        freq=20,
        callback=position_callback
    )

    time.sleep(1)

    if last_odom_time == 0:
        print("[WARN] No odometry data received yet.")
    else:
        print(
            f"[init] Start odometry: "
            f"x={current_x:.3f}, y={current_y:.3f}, yaw={current_yaw:.2f}"
        )

    print("[init] Starting camera stream...")
    ep_camera.start_video_stream(
        display=False,
        resolution=camera.STREAM_360P
    )

    last_map_print = time.time()
    last_map_save = time.time()

    try:
        inc = 0
        while True:
            # ----------------------------------------------------
            # Read newest camera frame
            # ----------------------------------------------------
            try:
                frame = ep_camera.read_cv2_image(
                    strategy="newest",
                    timeout=0.5
                )

                inc += 1

                if inc % 100 == 0:
                    print(f"[debug] Frame read successfully at {time.time():.2f}s")
                    inc = 0

                    mapper.plot_robot_pos(current_x, current_y, current_yaw)



            except Empty:
                print("[WARN] No camera frame received.")
                continue

            except Exception as e:
                print(f"[WARN] Camera read error: {e}")
                continue

            if frame is None:
                continue

            # ----------------------------------------------------
            # 1. Explore decides motion
            # ----------------------------------------------------
            state, x_spd, y_spd, z_spd, explore_debug = explorer.update(frame)

            # ----------------------------------------------------
            # 2. Mapper observes the same frame
            # Mapper does NOT control motion.
            # ----------------------------------------------------
            tag_events = mapper.update(
                frame=frame,
                robot_x=current_x,
                robot_y=current_y,
                robot_yaw_deg=current_yaw
            )

            # ----------------------------------------------------
            # 3. Send motion command from explorer
            # ----------------------------------------------------
            ep_chassis.drive_speed(
                x=x_spd,
                y=y_spd,
                z=z_spd,
                timeout=0.1
            )

            # ----------------------------------------------------
            # 4. Show debug view
            # ----------------------------------------------------
            main_debug = draw_main_debug(
                frame=explore_debug,
                state=state,
                tag_events=tag_events,

                
                mapper=mapper
            )

            cv2.imshow("Explore + Mapping", main_debug)

            # ----------------------------------------------------
            # 5. Print map every 5 seconds
            # ----------------------------------------------------
            if time.time() - last_map_print > 5:
                mapper.print_map()
                last_map_print = time.time()

            # ----------------------------------------------------
            # 6. Autosave map every 15 seconds
            # ----------------------------------------------------
            if time.time() - last_map_save > 15:
                mapper.save_map_plot("generated_map_autosave.png")
                last_map_save = time.time()

            # ----------------------------------------------------
            # Keyboard controls
            # ----------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[system] q pressed. Exiting.")
                break

            if key == ord("m"):
                mapper.print_map()

            if key == ord("s"):
                mapper.save_map_plot("generated_map_manual.png")

    except KeyboardInterrupt:
        print("\n[system] Keyboard interrupt received.")

    finally:
        print("\n[system] Stopping robot...")
        ep_chassis.drive_speed(x=0, y=0, z=0)

        print("[system] Saving final map...")
        mapper.print_map()
        mapper.save_map_plot("generated_map_final.png")

        print("[system] Cleaning up...")
        try:
            ep_chassis.unsub_position()
        except Exception as e:
            print(f"[WARN] Could not unsubscribe odometry: {e}")

        try:
            ep_camera.stop_video_stream()
        except Exception as e:
            print(f"[WARN] Could not stop camera stream: {e}")

        ep_robot.close()
        cv2.destroyAllWindows()

        print("[system] Robot disconnected cleanly.")


if __name__ == "__main__":
    main()

