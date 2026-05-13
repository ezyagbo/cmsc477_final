

import cv2
import numpy as np
import robomaster
from robomaster import robot, camera
from ultralytics import YOLO
import time
import threading
from queue import Empty
import pupil_apriltags
from approach_tag import AprilTagDetector

# ─── Tuned HSV ranges from actual blue tape sample ─────────────────────
BLUE_HSV_LOWER = np.array([100, 60,  80])
BLUE_HSV_UPPER = np.array([120, 255, 255])


# ─── Tunable parameters ────────────────────────────────────────────────
DANGER_ZONE_RATIO       = 0.35   # bottom 35% of frame is the proximity region
PIXEL_DENSITY_THRESHOLD = 0.04   # 4% blue pixels in danger zone → boundary
LINEAR_SPEED            = 0.3    # m/s forward (EP uses -3.5 to 3.5)
TURN_SPEED              = 30     # deg/s rotation (EP uses -600 to 600)
LOOP_HZ                 = 20     # control loop frequency
# ─── QR CODE ARRAYS ────────────────────────────────────────────────────

# Block Dock 1(Big/Small): 45/41
# Block Dock 2 (Big/Small): 34/38
# Charging Station: 10/8

LARGE_GOALS = [45, 34]
SMALL_GOALS = [41, 38]
CHARGING = [10, 8]
APRIL_TAG_DIST = 0.4
IR_DISTANCE_THRESHOLD = 0.4   # meters — back away if IR sees anything closer
BOX_HEIGHT_THRESHOLD   = 0.8   # box bbox height / frame height → trigger avoidance (close)
BLOCK_HEIGHT_THRESHOLD = 0.4  # trigger block avoidance at ~1 m distance (far)
BLOCK_CLASSES = {"large_brick", "small_brick", "large brick", "small brick"}

# ── Vision helpers ──────────────────────────────────────────────────────


def get_blue_mask(frame: np.ndarray) -> np.ndarray:
   """Return cleaned binary mask of blue tape pixels."""
   hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   mask   = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
   kernel = np.ones((5, 5), np.uint8)
   mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)
   mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
   return mask




def compute_boundary_error(frame: np.ndarray):
   """
   Returns (error, direction_bias):
     error          : blue pixel density in danger zone [0.0–1.0]
     direction_bias : >0 → tape more on right, <0 → tape more on left
   """
   h, w       = frame.shape[:2]
   danger_top = int(h * (1 - DANGER_ZONE_RATIO))


   mask        = get_blue_mask(frame)
   danger_zone = mask[danger_top:, :]


   blue_pixels  = np.count_nonzero(danger_zone)
   total_pixels = danger_zone.size
   error        = blue_pixels / total_pixels


   left_blue  = np.count_nonzero(danger_zone[:, :w // 2])
   right_blue = np.count_nonzero(danger_zone[:, w // 2:])
   bias       = (right_blue - left_blue) / (danger_zone[:, :w // 2].size)


   return error, bias


def detect_ir_obstacle(distance) -> bool:
    """Return True if IR sensor reading is within the avoidance threshold."""
    return distance is not None and 0 < distance <= IR_DISTANCE_THRESHOLD


def detect_box_obstacle(frame, model):
    """
    Run YOLO on frame and return (detected, bias) for the largest brown box.
    bias > 0 means the box is on the right side of the frame.
    """
    if model is None:
        return False, 0.0

    results = model(frame, verbose=False)
    h, w = frame.shape[:2]

    best_box = None
    best_height = 0.0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_h = y2 - y1
            if box_h > best_height:
                best_height = box_h
                best_box = (x1, y1, x2, y2)

    if best_box is None or (best_height / h) < BOX_HEIGHT_THRESHOLD:
        return False, 0.0

    x1, y1, x2, y2 = best_box
    box_center_x = (x1 + x2) / 2.0
    bias = box_center_x - (w / 2.0)  # positive → box on right

    print(f"[YOLO] box detected (h_ratio={best_height/h:.2f}), bias={bias:.1f}")
    return True, bias


def detect_block_obstacle(frame, model):
    """
    Return (detected, bias) for the nearest brick/block in the frame.
    Uses a much lower height threshold than detect_box_obstacle so the
    robot starts turning away while the block is still far off.
    """
    if model is None:
        return False, 0.0

    results = model(frame, verbose=False)
    h, w = frame.shape[:2]

    best_box    = None
    best_height = 0.0

    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            class_name = names.get(cls_id, "").strip().lower()
            if class_name not in BLOCK_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_h = y2 - y1
            if box_h > best_height:
                best_height = box_h
                best_box    = (x1, y1, x2, y2)

    if best_box is None or (best_height / h) < BLOCK_HEIGHT_THRESHOLD:
        return False, 0.0

    x1, y1, x2, y2 = best_box
    bias = ((x1 + x2) / 2.0) - (w / 2.0)  # positive → block on right
    print(f"[YOLO BLOCK] block detected (h_ratio={best_height/h:.2f}), bias={bias:.1f}")
    return True, bias


def detect_apriltag_obstacle(frame):
    K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]])  # Camera focal length and center pixel
    marker_size_m = 0.153

    apriltag = AprilTagDetector()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = apriltag.find_tags(gray)

    if not detections:
        return False, 0.0

    # choose closest tag by image area (bigger = closer)
    tag = max(detections, key=lambda d: cv2.contourArea(d.corners.astype(int)))
    print(f"closest tag: {tag.tag_id}")

    tag_dist = np.linalg.norm(tag.pose_t)
    print(f"distance to tag: {tag_dist}")

    if tag_dist > APRIL_TAG_DIST:
        print(f"dist: {tag_dist}... no tag found")
        return False, 0
    
    frame_center_x = frame.shape[1] / 2.0
    tag_center_x = tag.center[0]

    bias = tag_center_x - frame_center_x  # positive means tag on right

    print(f"dist: {tag_dist}... tag found. bias: {bias}")
    return True, bias

# ── Debug overlay ───────────────────────────────────────────────────────


def draw_debug(frame, state, error, bias, turn_dir):
   out        = frame.copy()
   h, w       = out.shape[:2]
   danger_top = int(h * (1 - DANGER_ZONE_RATIO))


   mask = get_blue_mask(frame)
   out[mask > 0] = [0, 255, 0]  # highlight tape in green


   line_color = (0, 0, 255) if state == "TURN" else (0, 255, 0)
   cv2.line(out, (0, danger_top), (w, danger_top), line_color, 2)
   cv2.line(out, (w // 2, danger_top), (w // 2, h), (255, 255, 0), 1)


   turn_label = ("LEFT" if turn_dir > 0 else "RIGHT") if state == "TURN" else "—"
   for i, text in enumerate([
       f"State : {state}",
       f"Error : {error:.3f}  (thresh {PIXEL_DENSITY_THRESHOLD})",
       f"Bias  : {bias:+.3f}",
       f"Turn  : {turn_label}",
   ]):
       cv2.putText(out, text, (10, 30 + i * 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
   return out


# ── Boundary & Tag Controller ─────────────────────────────────────────────────


class ObstacleController:
   def __init__(self, yolo_model=None):
       self.state          = "EXPLORE"
       self.turn_direction = 1        # +1 = left, -1 = right
       self._lock          = threading.Lock()
       self._latest_frame  = None
       self._running       = False
       self._ir_distance   = None
       self._yolo_model    = yolo_model

   def on_ir_distance(self, sub_info):
       """Callback for IR distance sensor subscription."""
       distance = sub_info[0]
       with self._lock:
           self._ir_distance = distance

   def get_ir_distance(self):
       with self._lock:
           return self._ir_distance


   # Called by EP camera subscriber on every new frame
   def on_frame(self, frame):
       with self._lock:
           self._latest_frame = frame


   def get_latest_frame(self):
       with self._lock:
           return self._latest_frame
   
   def update(self, frame):
       """
       Process one frame → return (state, x_spd, y_spd, z_spd, debug_frame).
       EP drive_speed axes: x = forward/back, y = left/right, z = rotation (deg/s)
       """
       error, bias = compute_boundary_error(frame)
       tag_detected,   tag_bias   = detect_apriltag_obstacle(frame)
       box_detected,   box_bias   = detect_box_obstacle(frame, self._yolo_model)
       block_detected, block_bias = detect_block_obstacle(frame, self._yolo_model)
       # ir_dist = self.get_ir_distance()

       # # priority 1: IR obstacle within threshold — reverse away
       # if detect_ir_obstacle(ir_dist):
       #     print(f"[IR] obstacle at {ir_dist:.3f} m — backing up")
       #     self.state = "BACKUP"

       # priority 1: box very close
       if box_detected:
           print(f"[YOLO] box avoidance — turning {'left' if box_bias > 0 else 'right'}")
           self.state          = "TURN"
           self.turn_direction = -1 if box_bias > 0 else 1

       # priority 2: block detected far away — turn away early
       elif block_detected:
           print(f"[YOLO BLOCK] block avoidance — turning {'left' if block_bias > 0 else 'right'}")
           self.state          = "TURN"
           self.turn_direction = -1 if block_bias > 0 else 1

       # priority 3: avoid april tags
       elif tag_detected:
            self.state          = "TURN"
            self.turn_direction = -1 if tag_bias > 0 else 1

       elif self.state in ("EXPLORE", "BACKUP") and error > PIXEL_DENSITY_THRESHOLD:
           self.state          = "TURN"
           # turn AWAY from whichever side has more tape
           self.turn_direction = -1 if bias > 0 else 1

       # elif self.state == "BACKUP" and not detect_ir_obstacle(ir_dist):
       #     self.state = "EXPLORE"

       elif self.state == "TURN" and error < PIXEL_DENSITY_THRESHOLD * 0.5:
           self.state = "EXPLORE"


       if self.state == "EXPLORE":
           x_spd, y_spd, z_spd = LINEAR_SPEED, 0.0, 0.0
       elif self.state == "BACKUP":
           x_spd, y_spd, z_spd = -LINEAR_SPEED, 0.0, 0.0
       else:
           x_spd, y_spd, z_spd = 0.0, 0.0, TURN_SPEED * self.turn_direction


       debug = draw_debug(frame, self.state, error, bias, self.turn_direction)
       return self.state, x_spd, y_spd, z_spd, debug



# ── Main ────────────────────────────────────────────────────────────────


def main():
   robomaster.config.ROBOT_IP_STR = "192.168.50.113"
   ep_robot = robot.Robot()
   ep_robot.initialize(conn_type="sta")   # change to "ap" for direct Wi-Fi


   ep_chassis = ep_robot.chassis
   ep_camera  = ep_robot.camera
   # ep_sensor  = ep_robot.sensor_adaptor

   controller = ObstacleController()

   # ep_sensor.sub_distance(freq=20, callback=controller.on_ir_distance)

   # Start video stream — EP delivers 720p frames via callback
   ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)
   
   try:
        while True:
            try:
                frame = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
            except Empty:
                print("[WARN] No camera frame received, continuing...")
                continue
            except Exception as e:
                print(f"[WARN] Camera read error: {e}")
                continue


            if frame is None:
                continue              

            state, x_spd, y_spd, z_spd, debug = controller.update(frame)


            ep_chassis.drive_speed(
                x=x_spd,
                y=y_spd,
                z=z_spd,
                timeout=0.1
            )


            cv2.imshow("Boundary Controller", debug)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break



   except KeyboardInterrupt:
       print("Stopping...")


   finally:
       ep_chassis.drive_speed(x=0, y=0, z=0)   # full stop
       # ep_sensor.unsub_distance()
       ep_camera.stop_video_stream()
       ep_robot.close()
       cv2.destroyAllWindows()
       print("Robot disconnected cleanly.")




if __name__ == "__main__":
   main()

