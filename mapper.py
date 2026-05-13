import time
import json
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robomaster import robot, camera
from queue import Empty
from ultralytics import YOLO
import pupil_apriltags
from matplotlib.ticker import MultipleLocator
import robomaster




# ============================================================
# Robot settings
# ============================================================
ROBOT_IP = "192.168.50.121"


YOLO_MODEL_FILE = r"C://Users\\ezyag\\OneDrive\\Desktop\\UMD\\cmsc477\\Homework\\final\\best.pt"


MAP_JSON_FILE = "project3_mapper_map.json"
MAP_PLOT_FILE = "project3_mapper_map.png"




# ============================================================
# Motion settings
# ============================================================
DEFAULT_LINEAR_SPEED = 0.12      # m/s
DEFAULT_TURN_SPEED = 35          # deg/s
CONTROL_DT = 0.05                # loop delay




# ============================================================
# Scan settings
# ============================================================
DEFAULT_SCAN_DURATION = 1.5      # seconds


MIN_DETECTION_DISTANCE = 0.10    # meters
MAX_DETECTION_DISTANCE = 3.00    # meters
MIN_YOLO_CONF = 0.30
MAX_ABS_OBJECT_ANGLE_DEG = 60.0




# ============================================================
# AprilTag settings
# ============================================================
MARKER_SIZE_M = 0.153


# Approx camera matrix for RoboMaster 360p stream.
K = np.array([
    [314, 0, 320],
    [0, 314, 180],
    [0, 0, 1]
    # [130, 0, 180],
    # [0, 130, 120],
    # [0, 0, 1]
], dtype=np.float32)


FX = float(K[0, 0])
FY = float(K[1, 1])
CX = float(K[0, 2])
CY = float(K[1, 2])




# ============================================================
# Tower settings
# IMPORTANT:
# These names must match your YOLO model class names.
# The script also supports loose matching like "tall" or "small".
# ============================================================
TOWER_HEIGHTS_M = {
    "tall_tower": 0.189,
    "small_tower": 0.100,
}


TOWER_CLASS_ALIASES = {
    "tall tower": "tall_tower",
    "tall_tower": "tall_tower",
    "tall": "tall_tower",


    "small tower": "small_tower",
    "small_tower": "small_tower",
    "small": "small_tower",
}




# ============================================================
# Plot settings
# ============================================================
ARENA_SIZE_M = 3.0




# ============================================================
# Odometry / attitude globals
# ============================================================
current_x = 0.0
current_y = 0.0
current_yaw = 0.0


last_odom_time = 0.0
last_attitude_time = 0.0


origin_x = 0.0
origin_y = 0.0
origin_yaw = 0.0




def normalize_angle_deg(angle):
    """Keep angles in [-180, 180]."""
    return (angle + 180) % 360 - 180




def position_callback(position_info):
    """
    Position callback.


    Usually gives x, y, z position.
    We use x and y from here.
    We use yaw from attitude_callback.
    """
    global current_x, current_y, last_odom_time


    current_x = position_info[0]
    current_y = position_info[1]
    last_odom_time = time.time()




def attitude_callback(attitude_info):
    """
    Attitude callback.


    Usually gives yaw, pitch, roll in degrees.
    """
    global current_yaw, last_attitude_time


    yaw, pitch, roll = attitude_info
    current_yaw = yaw
    last_attitude_time = time.time()




def rel_pose():
    """
    Robot pose relative to where this script started.


    This keeps the map origin at the robot's starting pose.
    """
    robot_x = current_x - origin_x
    robot_y = current_y - origin_y
    robot_yaw = normalize_angle_deg(current_yaw - origin_yaw)


    return robot_x, robot_y, robot_yaw




# ============================================================
# AprilTag detector
# ============================================================
class AprilTagDetector:
    def __init__(self):
        self.camera_params = [
            K[0, 0],
            K[1, 1],
            K[0, 2],
            K[1, 2]
        ]


        self.detector = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=2
        )


    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=MARKER_SIZE_M
        )


        return detections




# ============================================================
# Map storage
# ============================================================
tag_map = {}
tower_map = {}




def classify_marker(tag_id):
    """
    Update these labels based on your actual competition tag IDs.
    """
    if tag_id == 34:
        return "charging_station"
    elif tag_id == 14:
        return "obstacle_1"
    elif tag_id == 10:
        return "obstacle_2"
    elif tag_id == 19:
        return "goal_or_marker_19"
    elif tag_id == 41:
        return "goal_or_marker_41"
    elif tag_id == 30:
        return "test_marker_30"
    elif tag_id == 31:
        return "test_marker_31"
    else:
        return "unknown_marker"




def estimate_world_position_from_angle_distance(distance_m, angle_rad):
    """
    Convert object distance and camera angle into map x,y.


    NOTE:
    If your map is mirrored left/right, change:
        robot_yaw_rad - angle_rad
    to:
        robot_yaw_rad + angle_rad
    """
    robot_x, robot_y, robot_yaw = rel_pose()
    robot_yaw_rad = math.radians(robot_yaw)


    heading_rad = robot_yaw_rad - angle_rad


    obj_x = robot_x + distance_m * math.cos(heading_rad)
    obj_y = robot_y + distance_m * math.sin(heading_rad)


    return obj_x, obj_y




def estimate_marker_world_position(detection):
    """
    Estimate AprilTag world/map position.


    AprilTag gives pose_t, so we use its 3D distance.
    Horizontal angle comes from pixel offset.
    """
    tag_dist = float(np.linalg.norm(detection.pose_t))


    tag_center_x = float(detection.center[0])
    tag_bias_px = tag_center_x - CX


    tag_angle_rad = math.atan2(tag_bias_px, FX)
    tag_angle_deg = math.degrees(tag_angle_rad)


    tag_x, tag_y = estimate_world_position_from_angle_distance(
        distance_m=tag_dist,
        angle_rad=tag_angle_rad
    )


    return tag_x, tag_y, tag_dist, tag_angle_deg, tag_bias_px




def detection_is_good(distance_m, angle_deg):
    """
    Reject bad detections before they enter the map average.
    """
    if distance_m < MIN_DETECTION_DISTANCE:
        return False


    if distance_m > MAX_DETECTION_DISTANCE:
        return False


    if abs(angle_deg) > MAX_ABS_OBJECT_ANGLE_DEG:
        return False


    return True




def update_tag_map(tag_id, tag_x, tag_y, tag_dist, tag_angle_deg, tag_bias_px):
    """
    Save or update AprilTag position using a running average.
    """
    tag_key = str(tag_id)
    marker_type = classify_marker(tag_id)


    robot_x, robot_y, robot_yaw = rel_pose()


    if tag_key not in tag_map:
        tag_map[tag_key] = {
            "id": tag_id,
            "type": marker_type,
            "x": tag_x,
            "y": tag_y,
            "count": 1,
            "last_distance": tag_dist,
            "last_angle_deg": tag_angle_deg,
            "last_bias_px": tag_bias_px,
            "last_robot_x": robot_x,
            "last_robot_y": robot_y,
            "last_robot_yaw": robot_yaw,
            "last_seen": time.time()
        }
    else:
        old = tag_map[tag_key]
        count = old["count"]
        new_count = count + 1


        old["x"] = (old["x"] * count + tag_x) / new_count
        old["y"] = (old["y"] * count + tag_y) / new_count
        old["count"] = new_count


        old["last_distance"] = tag_dist
        old["last_angle_deg"] = tag_angle_deg
        old["last_bias_px"] = tag_bias_px
        old["last_robot_x"] = robot_x
        old["last_robot_y"] = robot_y
        old["last_robot_yaw"] = robot_yaw
        old["last_seen"] = time.time()


    data = tag_map[tag_key]


    print(
        f"[TAG] id={tag_id:>2} "
        f"type={marker_type:<18} "
        f"map=({data['x']:.3f}, {data['y']:.3f}) "
        f"dist={tag_dist:.3f} m "
        f"angle={tag_angle_deg:+.1f} deg "
        f"count={data['count']}"
    )




def normalize_yolo_class_name(class_name):
    """
    Convert YOLO class name into our internal tower name.
    """
    name = class_name.strip().lower().replace("-", "_")


    if name in TOWER_CLASS_ALIASES:
        return TOWER_CLASS_ALIASES[name]


    name_space = name.replace("_", " ")
    if name_space in TOWER_CLASS_ALIASES:
        return TOWER_CLASS_ALIASES[name_space]


    if "tall" in name:
        return "tall_tower"


    if "small" in name or "short" in name:
        return "small_tower"


    return None




def estimate_tower_from_box(box_xyxy, tower_type):
    """
    Estimate tower map position from YOLO bounding box.


    YOLO gives image box, not real distance.
    We estimate distance using known tower height:


        distance = real_height * FY / pixel_height


    This works best when:
    - tower is upright
    - full tower is visible
    - camera sees tower mostly from the front
    """
    x1, y1, x2, y2 = box_xyxy


    box_center_x = (x1 + x2) / 2.0
    box_height_px = max(1.0, y2 - y1)


    real_height_m = TOWER_HEIGHTS_M[tower_type]


    distance_m = (real_height_m * FY) / box_height_px


    bias_px = box_center_x - CX
    angle_rad = math.atan2(bias_px, FX)
    angle_deg = math.degrees(angle_rad)


    obj_x, obj_y = estimate_world_position_from_angle_distance(
        distance_m=distance_m,
        angle_rad=angle_rad
    )


    return obj_x, obj_y, distance_m, angle_deg, bias_px, box_height_px




def update_tower_map(tower_type, obj_x, obj_y, distance_m, angle_deg, conf, box_height_px):
    """
    Save or update tower position using a running average.


    Since there may be multiple towers of the same type, we cluster detections.
    If a new detection is close to an existing tower, update that tower.
    If not, create a new tower entry.
    """
    robot_x, robot_y, robot_yaw = rel_pose()


    # Cluster threshold.
    # If detections are within this distance, treat them as the same tower.
    MERGE_DISTANCE_M = 0.5


    best_key = None
    best_dist = None


    for key, data in tower_map.items():
        if data["type"] != tower_type:
            continue


        dx = obj_x - data["x"]
        dy = obj_y - data["y"]
        d = math.hypot(dx, dy)


        if best_dist is None or d < best_dist:
            best_dist = d
            best_key = key


    if best_key is not None and best_dist is not None and best_dist < MERGE_DISTANCE_M:
        old = tower_map[best_key]
        count = old["count"]
        new_count = count + 1


        old["x"] = (old["x"] * count + obj_x) / new_count
        old["y"] = (old["y"] * count + obj_y) / new_count
        old["count"] = new_count
        old["last_distance"] = distance_m
        old["last_angle_deg"] = angle_deg
        old["last_conf"] = conf
        old["last_box_height_px"] = box_height_px
        old["last_robot_x"] = robot_x
        old["last_robot_y"] = robot_y
        old["last_robot_yaw"] = robot_yaw
        old["last_seen"] = time.time()


        key = best_key


    else:
        same_type_count = sum(1 for data in tower_map.values() if data["type"] == tower_type)
        key = f"{tower_type}_{same_type_count + 1}"


        tower_map[key] = {
            "id": key,
            "type": tower_type,
            "x": obj_x,
            "y": obj_y,
            "count": 1,
            "last_distance": distance_m,
            "last_angle_deg": angle_deg,
            "last_conf": conf,
            "last_box_height_px": box_height_px,
            "last_robot_x": robot_x,
            "last_robot_y": robot_y,
            "last_robot_yaw": robot_yaw,
            "last_seen": time.time()
        }


    data = tower_map[key]


    print(
        f"[TOWER] {key:<14} "
        f"type={tower_type:<11} "
        f"map=({data['x']:.3f}, {data['y']:.3f}) "
        f"dist={distance_m:.3f} m "
        f"angle={angle_deg:+.1f} deg "
        f"conf={conf:.2f} "
        f"count={data['count']}"
    )




def detect_and_map_towers(frame, yolo_model):
    """
    Run YOLO and update tower_map.
    """
    tower_events = []


    results = yolo_model(frame, verbose=False)


    for result in results:
        names = result.names


        if result.boxes is None:
            continue


        for box in result.boxes:
            conf = float(box.conf[0])


            if conf < MIN_YOLO_CONF:
                continue


            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))


            tower_type = normalize_yolo_class_name(class_name)
            print(f"[YOLO RAW] class={class_name} conf={conf:.2f}")


            if tower_type is None:
                continue


            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            obj_x, obj_y, distance_m, angle_deg, bias_px, box_height_px = estimate_tower_from_box(
                box_xyxy=xyxy,
                tower_type=tower_type
            )


            if not detection_is_good(distance_m, angle_deg):
                print(
                    f"[SKIP TOWER] type={tower_type} "
                    f"dist={distance_m:.2f} "
                    f"angle={angle_deg:+.1f} "
                    f"conf={conf:.2f}"
                )
                continue


            update_tower_map(
                tower_type=tower_type,
                obj_x=obj_x,
                obj_y=obj_y,
                distance_m=distance_m,
                angle_deg=angle_deg,
                conf=conf,
                box_height_px=box_height_px
            )


            tower_events.append({
                "type": tower_type,
                "x": obj_x,
                "y": obj_y,
                "distance": distance_m,
                "angle_deg": angle_deg,
                "conf": conf,
                "box": xyxy.tolist()
            })


    return tower_events, results




# ============================================================
# Camera debug
# ============================================================
def draw_debug(frame, tag_detections, tower_events, yolo_results, scan_label):
    out = frame.copy()


    robot_x, robot_y, robot_yaw = rel_pose()


    lines = [
        f"{scan_label}",
        f"Odom rel: x={robot_x:.2f}, y={robot_y:.2f}, yaw={robot_yaw:.1f}",
        f"Tags mapped: {len(tag_map)}",
        f"Towers mapped: {len(tower_map)}",
        "Press q to stop early"
    ]


    for i, text in enumerate(lines):
        cv2.putText(
            out,
            text,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # Draw AprilTags
    for detection in tag_detections:
        tag_id = int(detection.tag_id)
        corners = detection.corners.astype(int)
        center = tuple(detection.center.astype(int))


        cv2.polylines(out, [corners.reshape((-1, 1, 2))], True, (0, 255, 0), 2)
        cv2.circle(out, center, 4, (0, 0, 255), -1)
        cv2.putText(
            out,
            f"Tag {tag_id}",
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Draw YOLO boxes
    for result in yolo_results:
        names = result.names


        if result.boxes is None:
            continue


        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, str(cls_id))
            tower_type = normalize_yolo_class_name(class_name)


            if tower_type is None or conf < MIN_YOLO_CONF:
                continue


            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int)


            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                out,
                f"{tower_type} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )


    return out




def scan_stationary(ep_camera, tag_detector, yolo_model, duration=DEFAULT_SCAN_DURATION, label="stationary scan"):
    """
    Scan only while the robot is stopped.


    This function updates:
    - tag_map from AprilTags
    - tower_map from YOLO tower detections
    """
    print(f"\n[SCAN] {label}")
    print(f"       duration={duration:.2f}s")
    print("       robot is stopped, mapping enabled")


    start_time = time.time()


    while time.time() - start_time < duration:
        try:
            frame = ep_camera.read_cv2_image(
                strategy="newest",
                timeout=0.3
            )


        except Empty:
            print("[WARN] No camera frame.")
            time.sleep(CONTROL_DT)
            continue


        except Exception as e:
            print(f"[WARN] Camera read error: {e}")
            time.sleep(CONTROL_DT)
            continue


        if frame is None:
            time.sleep(CONTROL_DT)
            continue


        # AprilTag mapping
        tag_detections = tag_detector.detect(frame)


        for detection in tag_detections:
            if detection.pose_t is None:
                continue


            tag_id = int(detection.tag_id)


            tag_x, tag_y, tag_dist, tag_angle_deg, tag_bias_px = estimate_marker_world_position(
                detection
            )


            if not detection_is_good(tag_dist, tag_angle_deg):
                print(
                    f"[SKIP TAG] id={tag_id} "
                    f"dist={tag_dist:.2f} "
                    f"angle={tag_angle_deg:+.1f}"
                )
                continue


            update_tag_map(
                tag_id=tag_id,
                tag_x=tag_x,
                tag_y=tag_y,
                tag_dist=tag_dist,
                tag_angle_deg=tag_angle_deg,
                tag_bias_px=tag_bias_px
            )


        # YOLO tower mapping
        tower_events, yolo_results = detect_and_map_towers(frame, yolo_model)


        debug_frame = draw_debug(
            frame=frame,
            tag_detections=tag_detections,
            tower_events=tower_events,
            yolo_results=yolo_results,
            scan_label=label
        )


        cv2.imshow("Project 3 Mapper", debug_frame)


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):
            print("[SCAN] q pressed. Stopping route early.")
            return False


        time.sleep(CONTROL_DT)


    return True




# ============================================================
# Movement helpers
# No mapping happens in these functions.
# ============================================================
def stop_robot(ep_chassis):
    try:
        ep_chassis.drive_speed(x=0, y=0, z=0)
    except Exception:
        pass


    time.sleep(0.25)




def move_for_duration(ep_chassis, x_speed, y_speed, z_speed, duration, label):
    """
    Move robot for a time duration.


    No camera scanning.
    No map updates.
    """
    print(f"\n[MOVE] {label}")
    print(
        f"       x_speed={x_speed:.3f}, "
        f"y_speed={y_speed:.3f}, "
        f"z_speed={z_speed:.1f}, "
        f"duration={duration:.2f}s"
    )


    start_time = time.time()


    while time.time() - start_time < duration:
        ep_chassis.drive_speed(
            x=x_speed,
            y=y_speed,
            z=z_speed,
            timeout=0.1
        )


        time.sleep(CONTROL_DT)


    stop_robot(ep_chassis)


    robot_x, robot_y, robot_yaw = rel_pose()
    print(f"[ODOM] after move: x={robot_x:.3f}, y={robot_y:.3f}, yaw={robot_yaw:.2f}")




def forward(ep_chassis, distance_m, speed=DEFAULT_LINEAR_SPEED):
    if speed == 0:
        print("[WARN] forward speed is 0. Skipping.")
        return


    duration = abs(distance_m / speed)
    x_speed = math.copysign(abs(speed), distance_m)


    move_for_duration(
        ep_chassis=ep_chassis,
        x_speed=x_speed,
        y_speed=0,
        z_speed=0,
        duration=duration,
        label=f"Forward {distance_m:.2f} m"
    )




def slide(ep_chassis, distance_m, speed=DEFAULT_LINEAR_SPEED):
    if speed == 0:
        print("[WARN] slide speed is 0. Skipping.")
        return


    duration = abs(distance_m / speed)
    y_speed = math.copysign(abs(speed), distance_m)


    move_for_duration(
        ep_chassis=ep_chassis,
        x_speed=0,
        y_speed=y_speed,
        z_speed=0,
        duration=duration,
        label=f"Slide {distance_m:.2f} m"
    )




def turn(ep_chassis, angle_deg, speed=DEFAULT_TURN_SPEED):
    if speed == 0:
        print("[WARN] turn speed is 0. Skipping.")
        return


    duration = abs(angle_deg / speed)
    z_speed = math.copysign(abs(speed), angle_deg)


    move_for_duration(
        ep_chassis=ep_chassis,
        x_speed=0,
        y_speed=0,
        z_speed=z_speed,
        duration=duration,
        label=f"Turn {angle_deg:.1f} deg"
    )




# ============================================================
# Hardcoded route
# ============================================================
def run_mapping_route(ep_chassis, ep_camera, tag_detector, yolo_model):
    """
    Edit this route.


    Rule:
        move/turn/slide = no mapping
        scan_stationary = mapping


    Since your towers are clustered in one area, the goal is to:
    - drive to viewpoints that can see the cluster
    - stop
    - scan
    - rotate/shift to improve angles
    """
    print("\n[ROUTE] Starting Project 3 mapping route")


    if not scan_stationary(ep_camera, tag_detector, yolo_model, duration=2.0, label="scan at start"):
        return


    # --------------------------------------------------------
    # Example route.
    # Tune these to your arena.
    # --------------------------------------------------------


    forward(ep_chassis, 1.5)
    if not scan_stationary(ep_camera, tag_detector, yolo_model, duration=1.5, label="scan after forward 1.5m"):
        return
    
    turn(ep_chassis, 90)
    if not scan_stationary(ep_camera, tag_detector, yolo_model, duration=1.5, label="scan after turn +30"):
        return
    


    # Add more route steps here if needed:
    #
    # slide(ep_chassis, 0.20)
    # scan_stationary(ep_camera, tag_detector, yolo_model, duration=1.5, label="scan from right side")
    #
    # slide(ep_chassis, -0.40)
    # scan_stationary(ep_camera, tag_detector, yolo_model, duration=1.5, label="scan from left side")


    print("\n[ROUTE] Mapping route complete")




# ============================================================
# Save / print map
# ============================================================
def make_world_map():
    robot_x, robot_y, robot_yaw = rel_pose()


    return {
        "created_time": time.time(),
        "robot_final_pose": {
            "x": robot_x,
            "y": robot_y,
            "yaw_deg": robot_yaw
        },
        "tags": tag_map,
        "towers": tower_map
    }




def save_map_json(filename=MAP_JSON_FILE):
    world_map = make_world_map()


    with open(filename, "w") as f:
        json.dump(world_map, f, indent=4)


    print(f"\n[MAP] Saved JSON map to {filename}")




def save_map_plot(filename=MAP_PLOT_FILE):
    fig, ax = plt.subplots(figsize=(8, 8))


    ax.set_title("Project 3 Map: Tags + Towers")
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")


    ax.set_xlim(-1.0, ARENA_SIZE_M + 1.0)
    ax.set_ylim(-1.0, ARENA_SIZE_M + 1.0)


    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))


    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))


    ax.grid(which="major", linewidth=0.7)
    ax.grid(which="minor", linewidth=0.3, alpha=0.6)


    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="y", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="y", which="minor", labelleft=False)


    ax.set_aspect("equal", adjustable="box")


    robot_x, robot_y, robot_yaw = rel_pose()


    ax.scatter(robot_x, robot_y, marker="o", s=100, color="gray", label="Robot final")
    ax.text(robot_x + 0.03, robot_y + 0.03, "Robot", fontsize=8)


    # Plot tags
    for tag_key, data in tag_map.items():
        tag_id = data["id"]
        marker_type = data["type"]
        x = data["x"]
        y = data["y"]


        ax.scatter(x, y, marker="x", s=120, label="AprilTag")
        ax.text(x + 0.03, y + 0.03, f"Tag {tag_id}: {marker_type}", fontsize=8)


    # Plot towers
    for tower_key, data in tower_map.items():
        tower_type = data["type"]
        x = data["x"]
        y = data["y"]


        if tower_type == "tall_tower":
            ax.scatter(x, y, marker="^", s=180, label="Tall tower")
        else:
            ax.scatter(x, y, marker="s", s=160, label="Small tower")


        ax.text(x + 0.03, y + 0.03, tower_key, fontsize=8)


    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))


    if unique:
        ax.legend(unique.values(), unique.keys(), loc="upper right")


    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


    print(f"[MAP] Saved plot to {filename}")




def print_final_map():
    print("\n========== FINAL PROJECT 3 MAP ==========")


    print("\n--- APRILTAGS ---")
    if not tag_map:
        print("No tags detected.")
    else:
        for tag_id, data in tag_map.items():
            print(
                f"Tag {tag_id}: "
                f"type={data['type']}, "
                f"x={data['x']:.3f}, "
                f"y={data['y']:.3f}, "
                f"count={data['count']}"
            )


    print("\n--- TOWERS ---")
    if not tower_map:
        print("No towers detected.")
    else:
        for tower_id, data in tower_map.items():
            print(
                f"{tower_id}: "
                f"type={data['type']}, "
                f"x={data['x']:.3f}, "
                f"y={data['y']:.3f}, "
                f"count={data['count']}, "
                f"last_conf={data['last_conf']:.2f}"
            )


    print("=========================================\n")




# ============================================================
# Main
# ============================================================
def main():
    global origin_x, origin_y, origin_yaw


    robomaster.config.ROBOT_IP_STR = ROBOT_IP


    print("[init] Loading YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_FILE)


    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="sta")


    ep_chassis = ep_robot.chassis
    ep_camera = ep_robot.camera


    tag_detector = AprilTagDetector()


    print("[init] Subscribing to odometry...")
    ep_chassis.sub_position(
        cs=1,
        freq=20,
        callback=position_callback
    )


    print("[init] Subscribing to attitude...")
    ep_chassis.sub_attitude(
        freq=20,
        callback=attitude_callback
    )


    time.sleep(1)


    origin_x = current_x
    origin_y = current_y
    origin_yaw = current_yaw


    print(
        f"[init] Origin set: "
        f"x={origin_x:.3f}, y={origin_y:.3f}, yaw={origin_yaw:.2f}"
    )


    print("[init] Starting camera stream...")
    ep_camera.start_video_stream(
        display=False,
        resolution=camera.STREAM_360P
    )


    try:
        run_mapping_route(
            ep_chassis=ep_chassis,
            ep_camera=ep_camera,
            tag_detector=tag_detector,
            yolo_model=yolo_model
        )


        print_final_map()
        save_map_json()
        save_map_plot()


    except KeyboardInterrupt:
        print("\n[system] Keyboard interrupt received.")


    finally:
        print("\n[system] Cleaning up...")


        try:
            stop_robot(ep_chassis)
        except Exception:
            pass


        try:
            ep_chassis.unsub_position()
        except Exception:
            pass


        try:
            ep_chassis.unsub_attitude()
        except Exception:
            pass


        try:
            ep_camera.stop_video_stream()
        except Exception:
            pass


        try:
            ep_robot.close()
        except Exception:
            pass


        cv2.destroyAllWindows()


        print("[system] Done.")




if __name__ == "__main__":
    main()



