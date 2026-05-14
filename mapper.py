



import time
import json
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import robomaster
from robomaster import robot, camera
from queue import Empty
from ultralytics import YOLO
import pupil_apriltags
from matplotlib.ticker import MultipleLocator
# from test_explore_and_charge import main as explore_and_charge_main




# ============================================================
# Robot settings
# ============================================================
ROBOT_IP = "192.168.50.121"
YOLO_MODEL_FILE = r"C:\\Users\\ezyag\\OneDrive\\Desktop\\UMD\\cmsc477\\Homework\\final\\best.pt"
MAP_JSON_FILE = "project3_mapper_map.json"
MAP_PLOT_FILE = "project3_mapper_map.png"


# ============================================================
# Motion settings
# ============================================================
DEFAULT_LINEAR_SPEED = 0.32
DEFAULT_TURN_SPEED = 45
CONTROL_DT = 0.05


# ============================================================
# Scan settings
# ============================================================
DEFAULT_SCAN_DURATION = 1.5
MIN_DETECTION_DISTANCE = 0.10
MAX_DETECTION_DISTANCE = 3.00
MIN_YOLO_CONF = 0.30
MAX_ABS_OBJECT_ANGLE_DEG = 60.0


# ============================================================
# AprilTag / Camera settings
# ============================================================
MARKER_SIZE_M = 0.153


K = np.array([
    [314, 0, 320],
    [0, 314, 180],
    [0, 0,   1]
], dtype=np.float32)


FX = float(K[0, 0])
FY = float(K[1, 1])
CX = float(K[0, 2])
CY = float(K[1, 2])


# ============================================================
# LEGO brick / tower settings
# ============================================================
TOWER_HEIGHTS_M = {
    "tall_tower": 0.189,
    "small_tower": 0.100,
}


TOWER_CLASS_ALIASES = {
    "tall tower":  "tall_tower",
    "tall_tower":  "tall_tower",
    "tall":        "tall_tower",
    "small tower": "small_tower",
    "small_tower": "small_tower",
    "small":       "small_tower",
}


ARENA_SIZE_M = 3.0




# ============================================================
# Odometry globals
# ============================================================
current_x   = 0.0
current_y   = 0.0
current_yaw = 0.0


origin_x   = 0.0
origin_y   = 0.0
origin_yaw = 0.0


INC = 0
ROBO_POSE     = []
ROBO_ATTITUDE = []




def normalize_angle_deg(angle):
    return (angle + 180) % 360 - 180




def position_callback(position_info):
    global current_x, current_y, INC
    current_x = position_info[0]
    current_y = position_info[1]
    INC += 1
    if INC % 10 == 0:
        rx, ry, ryaw = rel_pose()
        ROBO_POSE.append((rx, ry))
        ROBO_ATTITUDE.append(ryaw)




def attitude_callback(attitude_info):
    global current_yaw
    yaw, pitch, roll = attitude_info
    current_yaw = yaw




def rel_pose():
    rx   = current_x   - origin_x
    ry   = current_y   - origin_y
    ryaw = normalize_angle_deg(current_yaw - origin_yaw)
    return rx, ry, ryaw




# ============================================================
# TRANSFORMATION MATRIX
# ============================================================


def make_transform(x, y, yaw_deg):
    """
    Build 3x3 SE(2) transform: robot frame → world frame.


        T = [ cos(θ)  -sin(θ)   x ]
            [ sin(θ)   cos(θ)   y ]
            [   0        0      1 ]
    """
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ], dtype=np.float64)




def get_robot_transform():
    """Current robot-to-world transform from live odometry."""
    rx, ry, ryaw = rel_pose()
    return make_transform(rx, ry, ryaw)




def robot_point_to_world(dx_robot, dy_robot):
    """
    Transform a point in robot frame into world frame.


    dx_robot = forward distance (m)
    dy_robot = lateral distance (m, left = positive)
    """
    T = get_robot_transform()
    p_robot = np.array([dx_robot, dy_robot, 1.0])
    p_world = T @ p_robot
    return float(p_world[0]), float(p_world[1])




def distance_angle_to_robot_frame(distance_m, angle_rad):
    """
    Convert polar camera detection into robot-frame cartesian.


    dx = forward component
    dy = lateral component
    """
    dx = distance_m * math.cos(angle_rad)
    dy = distance_m * math.sin(angle_rad)
    return dx, dy




# ============================================================
# AprilTag detector
# ============================================================
class AprilTagDetector:
    def __init__(self):
        self.camera_params = [K[0,0], K[1,1], K[0,2], K[1,2]]
        self.detector = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=2
        )


    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=MARKER_SIZE_M
        )




# ============================================================
# Map storage
# ============================================================
tag_map   = {}   # keyed by str(tag_id)
tower_map = {}   # only ONE entry: "block_cluster"




# ============================================================
# AprilTag world position estimation
# ============================================================


def estimate_marker_world_position(detection):
    """
    Pipeline:
        pose_t → distance
        pixel offset → angle
        distance + angle → robot frame (dx, dy)
        robot frame → world frame via T
    """
    tag_dist  = float(np.linalg.norm(detection.pose_t))
    bias_px   = float(detection.center[0]) - CX
    angle_rad = math.atan2(bias_px, FX)
    angle_deg = math.degrees(angle_rad)


    dx, dy       = distance_angle_to_robot_frame(tag_dist, angle_rad)
    tag_x, tag_y = robot_point_to_world(dx, dy)


    return tag_x, tag_y, tag_dist, angle_deg, bias_px




def detection_is_good(distance_m, angle_deg):
    if distance_m < MIN_DETECTION_DISTANCE:       return False
    if distance_m > MAX_DETECTION_DISTANCE:       return False
    if abs(angle_deg) > MAX_ABS_OBJECT_ANGLE_DEG: return False
    return True




def update_tag_map(tag_id, tag_x, tag_y, tag_dist, angle_deg, bias_px):
    """Running-average world position for each tag ID."""
    key   = str(tag_id)
    label = f"obstacle_tag_{tag_id}"


    if key not in tag_map:
        tag_map[key] = {
            "id":    tag_id,
            "type":  label,
            "x":     tag_x,
            "y":     tag_y,
            "count": 1,
            "last_distance": tag_dist,
            "last_angle_deg": angle_deg,
            "last_bias_px":   bias_px,
            "last_seen": time.time()
        }
    else:
        old       = tag_map[key]
        count     = old["count"]
        new_count = count + 1
        old["x"]     = (old["x"] * count + tag_x) / new_count
        old["y"]     = (old["y"] * count + tag_y) / new_count
        old["count"] = new_count
        old["last_distance"]  = tag_dist
        old["last_angle_deg"] = angle_deg
        old["last_bias_px"]   = bias_px
        old["last_seen"]      = time.time()


    data = tag_map[key]
    print(
        f"[TAG] {label:<22} "
        f"map=({data['x']:.3f}, {data['y']:.3f})  "
        f"dist={tag_dist:.3f}m  "
        f"angle={angle_deg:+.1f}°  "
        f"count={data['count']}"
    )




# ============================================================
# LEGO block cluster — lock on first highest-conf detection
# ============================================================


def normalize_yolo_class_name(class_name):
    name = class_name.strip().lower().replace("-", "_")
    if name in TOWER_CLASS_ALIASES:
        return TOWER_CLASS_ALIASES[name]
    if name.replace("_", " ") in TOWER_CLASS_ALIASES:
        return TOWER_CLASS_ALIASES[name.replace("_", " ")]
    if "tall"  in name: return "tall_tower"
    if "small" in name or "short" in name: return "small_tower"
    return None




def estimate_block_world_position(box_xyxy, tower_type):
    """
    Pinhole distance from bounding box height, then transform to world.
    """
    x1, y1, x2, y2  = box_xyxy
    box_center_x     = (x1 + x2) / 2.0
    box_height_px    = max(1.0, y2 - y1)
    real_height_m    = TOWER_HEIGHTS_M[tower_type]


    distance_m = (real_height_m * FY) / box_height_px
    bias_px    = box_center_x - CX
    angle_rad  = math.atan2(bias_px, FX)
    angle_deg  = math.degrees(angle_rad)


    dx, dy = distance_angle_to_robot_frame(distance_m, angle_rad)
    wx, wy = robot_point_to_world(dx, dy)


    return wx, wy, distance_m, angle_deg, bias_px, box_height_px




def detect_and_map_blocks(frame, yolo_model):
    """
    Lock block_cluster once on the single highest-confidence YOLO detection.
    Returns (event_list, yolo_results) always — never None.
    """
    # Always run YOLO so we can draw boxes in debug overlay
    results = yolo_model(frame, verbose=False)


    # Already locked — return results for overlay only
    if "block_cluster" in tower_map:
        return [], results


    # Collect valid candidates
    candidates = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf      = float(box.conf[0])
            cls_id    = int(box.cls[0])
            class_name = result.names.get(cls_id, "")
            tower_type = normalize_yolo_class_name(class_name)


            if tower_type is None or conf < MIN_YOLO_CONF:
                continue


            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            candidates.append({
                "type": tower_type,
                "conf": conf,
                "box":  xyxy
            })


    if not candidates:
        return [], results


    # Pick single best by confidence
    best = max(candidates, key=lambda c: c["conf"])


    wx, wy, dist, angle_deg, bias_px, box_h = estimate_block_world_position(
        best["box"], best["type"]
    )


    if not detection_is_good(dist, angle_deg):
        print(
            f"[BLOCK SKIP] dist={dist:.2f}m  angle={angle_deg:+.1f}°  "
            f"conf={best['conf']:.2f} — failed quality check"
        )
        return [], results


    rx, ry, ryaw = rel_pose()
    tower_map["block_cluster"] = {
        "type":           best["type"],
        "x":              wx,
        "y":              wy,
        "conf":           best["conf"],
        "distance":       dist,
        "angle_deg":      angle_deg,
        "box_height_px":  box_h,
        "robot_x":        rx,
        "robot_y":        ry,
        "robot_yaw":      ryaw,
        "locked_time":    time.time()
    }


    print(
        f"[BLOCK LOCKED] type={best['type']}  "
        f"map=({wx:.3f}, {wy:.3f})  "
        f"dist={dist:.3f}m  "
        f"angle={angle_deg:+.1f}°  "
        f"conf={best['conf']:.2f}"
    )


    event = {"type": best["type"], "x": wx, "y": wy,
             "distance": dist, "conf": best["conf"]}
    return [event], results




# ============================================================
# Camera debug overlay
# ============================================================
def draw_debug(frame, tag_detections, tower_events, yolo_results, scan_label):
    out = frame.copy()
    rx, ry, ryaw = rel_pose()


    lines = [
        scan_label,
        f"World pose: x={rx:.2f}  y={ry:.2f}  yaw={ryaw:.1f}",
        f"Tags mapped: {len(tag_map)}",
        f"Block locked: {'YES' if tower_map else 'NO'}",
        "Press q to quit"
    ]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 30 + i*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    for det in tag_detections:
        corners = det.corners.astype(int)
        center  = tuple(det.center.astype(int))
        cv2.polylines(out, [corners.reshape(-1,1,2)], True, (0,255,0), 2)
        cv2.circle(out, center, 4, (0,0,255), -1)
        cv2.putText(out, f"Tag {int(det.tag_id)}", (center[0]+10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)


    for result in yolo_results:
        if result.boxes is None: continue
        for box in result.boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            name = normalize_yolo_class_name(result.names.get(cls, ""))
            if name is None or conf < MIN_YOLO_CONF: continue
            x1,y1,x2,y2 = box.xyxy[0].detach().cpu().numpy().astype(int)
            cv2.rectangle(out, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, max(20,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    return out




# ============================================================
# Stationary scan
# ============================================================
def scan_stationary(ep_camera, tag_detector, yolo_model,
                    duration=DEFAULT_SCAN_DURATION, label="scan"):
    print(f"\n[SCAN] {label}  duration={duration:.1f}s")
    start = time.time()


    while time.time() - start < duration:
        try:
            frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.3)
        except Empty:
            time.sleep(CONTROL_DT); continue
        except Exception as e:
            print(f"[WARN] {e}"); time.sleep(CONTROL_DT); continue
        if frame is None:
            time.sleep(CONTROL_DT); continue


        # AprilTags
        tag_detections = tag_detector.detect(frame)
        for det in tag_detections:
            if det.pose_t is None: continue
            tag_x, tag_y, tag_dist, angle_deg, bias_px = estimate_marker_world_position(det)
            if not detection_is_good(tag_dist, angle_deg): continue
            update_tag_map(int(det.tag_id), tag_x, tag_y, tag_dist, angle_deg, bias_px)


        # Block cluster
        tower_events, yolo_results = detect_and_map_blocks(frame, yolo_model)


        debug_frame = draw_debug(frame, tag_detections,
                                 tower_events, yolo_results, label)
        cv2.imshow("Project 3 Mapper", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False


        time.sleep(CONTROL_DT)


    return True




# ============================================================
# Movement helpers
# ============================================================
def stop_robot(ep_chassis):
    try: ep_chassis.drive_speed(x=0, y=0, z=0)
    except: pass
    time.sleep(0.25)




def move_for_duration(ep_chassis, x_speed, y_speed, z_speed, duration, label):
    print(f"\n[MOVE] {label}  x={x_speed:.3f} y={y_speed:.3f} z={z_speed:.1f} t={duration:.2f}s")
    start = time.time()
    while time.time() - start < duration:
        ep_chassis.drive_speed(x=x_speed, y=y_speed, z=z_speed, timeout=0.1)
        time.sleep(CONTROL_DT)
    stop_robot(ep_chassis)
    rx, ry, ryaw = rel_pose()
    print(f"[ODOM] x={rx:.3f} y={ry:.3f} yaw={ryaw:.2f}")




def forward(ep_chassis, distance_m, speed=DEFAULT_LINEAR_SPEED):
    duration = abs(distance_m / speed)
    move_for_duration(ep_chassis, math.copysign(speed, distance_m), 0, 0,
                      duration, f"Forward {distance_m:.2f}m")


def slide(ep_chassis, distance_m, speed=DEFAULT_LINEAR_SPEED):
    duration = abs(distance_m / speed)
    move_for_duration(ep_chassis, 0, math.copysign(speed, distance_m), 0,
                      duration, f"Slide {distance_m:.2f}m")


def turn(ep_chassis, angle_deg, speed=DEFAULT_TURN_SPEED):
    duration = abs(angle_deg / speed)
    move_for_duration(ep_chassis, 0, 0, math.copysign(speed, angle_deg),
                      duration, f"Turn {angle_deg:.1f}deg")




# ============================================================
# Route — edit distances/angles, structure is yours
# ============================================================
def run_mapping_route(ep_chassis, ep_camera, tag_detector, yolo_model):
    print("\n[ROUTE] Starting mapping route")


    # ── Waypoint 1 ──────────────────────────────────────────
    forward(ep_chassis, 1.2 )
    scan_stationary(ep_camera, tag_detector, yolo_model, duration=2.0, label="WP1 forward")


    for i in range(10):
        turn(ep_chassis, 45)
        scan_stationary(ep_camera, tag_detector, yolo_model,
                        duration=1.5, label=f"WP1 sweep {i+1}/10")
       
    turn(ep_chassis, 25)


    # ── Waypoint 2 ──────────────────────────────────────────
    forward(ep_chassis, 1.1)
    scan_stationary(ep_camera, tag_detector, yolo_model, duration=2.0, label="WP2 arrival")


    for i in range(10):
        turn(ep_chassis, 45)
        scan_stationary(ep_camera, tag_detector, yolo_model,
                        duration=1.5, label=f"WP2 sweep {i+1}/10")


    print("\n[ROUTE] Complete")




# ============================================================
# Save / print map
# ============================================================
def make_world_map():
    rx, ry, ryaw = rel_pose()
    return {
        "created_time": time.time(),
        "robot_final_pose": {"x": rx, "y": ry, "yaw_deg": ryaw},
        "tags":   tag_map,
        "towers": tower_map
    }


def save_map_json(filename=MAP_JSON_FILE):
    with open(filename, "w") as f:
        json.dump(make_world_map(), f, indent=4)
    print(f"[MAP] JSON saved → {filename}")




def save_map_plot(filename=MAP_PLOT_FILE):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Project 3 Map")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")


    # Expand to show all four quadrants
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_aspect("equal")


    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="major", linewidth=0.7)
    ax.grid(which="minor", linewidth=0.3, alpha=0.6)


    # Draw axis lines through origin
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)


    # Robot path — line only, no arrows
    if len(ROBO_POSE) >= 2:
        pa = np.array(ROBO_POSE)
        ax.plot(pa[:,0], pa[:,1], "-", color="gray", alpha=0.7, label="Robot path")


    # Robot final position dot
    rx, ry, _ = rel_pose()
    ax.scatter(rx, ry, s=100, color="gray", zorder=5, label="Robot final")


    # AprilTags
    for key, data in tag_map.items():
        ax.scatter(data["x"], data["y"], marker="x", s=150, color="red", zorder=5)
        ax.text(data["x"]+0.04, data["y"]+0.04,
                f"obstacle_tag_{data['id']}\n({data['x']:.2f},{data['y']:.2f})",
                fontsize=7, color="red")


    # Block cluster
    for key, data in tower_map.items():
        ax.scatter(data["x"], data["y"], marker="s", s=200, color="orange", zorder=5)
        ax.text(data["x"]+0.04, data["y"]+0.04,
                f"blocks\n({data['x']:.2f},{data['y']:.2f})",
                fontsize=7, color="orange")


    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels,handles)).values(),
              dict(zip(labels,handles)).keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"[MAP] Plot saved → {filename}")








def print_final_map():
    print("\n========== FINAL MAP ==========")
    print("--- TAGS ---")
    if not tag_map:
        print("  none detected")
    for k, v in tag_map.items():
        print(f"  obstacle_tag_{v['id']}: ({v['x']:.3f}, {v['y']:.3f})  count={v['count']}")
    print("--- BLOCK CLUSTER ---")
    if tower_map:
        d = tower_map["block_cluster"]
        print(f"  blocks: ({d['x']:.3f}, {d['y']:.3f})  conf={d['conf']:.2f}")
    else:
        print("  not detected")
    print("================================\n")




# ============================================================
# Main
# ============================================================
def main():
    global origin_x, origin_y, origin_yaw


    robomaster.config.ROBOT_IP_STR = ROBOT_IP


    print("[init] Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL_FILE)


    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")
    ep_chassis = ep_robot.chassis
    ep_camera  = ep_robot.camera
    tag_detector = AprilTagDetector()


    ep_chassis.sub_position(cs=1, freq=20, callback=position_callback)
    ep_chassis.sub_attitude(freq=20, callback=attitude_callback)


    # Wait long enough for callbacks to fire before capturing origin
    time.sleep(2)


    origin_x   = current_x
    origin_y   = current_y
    origin_yaw = current_yaw
    print(f"[init] Origin: x={origin_x:.3f} y={origin_y:.3f} yaw={origin_yaw:.2f}")


    ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)


    try:
        run_mapping_route(ep_chassis, ep_camera, tag_detector, yolo_model)
        print_final_map()
        save_map_json()
        save_map_plot()


    except KeyboardInterrupt:
        print("\n[system] Interrupted.")


    finally:
        stop_robot(ep_chassis)
        try: ep_chassis.unsub_position()
        except: pass
        try: ep_chassis.unsub_attitude()
        except: pass
        try: ep_camera.stop_video_stream()
        except: pass
        try: ep_robot.close()
        except: pass
        cv2.destroyAllWindows()
        print("[system] Done.")




if __name__ == "__main__":
    main()
    # explore_and_charge_main()
