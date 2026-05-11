import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pupil_apriltags


# ============================================================
# AprilTag IDs
# Update these if the real competition tags are different.
# ============================================================
LARGE_GOALS = [45, 19]
SMALL_GOALS = [41, 11]
CHARGING = [34, 38]
OBSTACLE_TAGS = [22, 14, 10, 8]


# ============================================================
# Mapping settings
# ============================================================
TAG_PING_DISTANCE = 0.25       # meters, log tag if seen within this distance
TAG_LOG_COOLDOWN = 2.0         # seconds, prevents logging same tag every frame
MAP_SIZE_METERS = 10.0          # arena is 6m x 6m
GRID_SPACING = 2            # 10 cm grid lines


class AprilTagDetector:
    """
    Small wrapper around pupil_apri
    ltags.

    This detects tags and estimates pose.
    """

    def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.153):
        self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
        self.marker_size_m = marker_size_m

        self.detector = pupil_apriltags.Detector(
            families=family,
            nthreads=threads
        )

    def find_tags(self, frame_gray):
        detections = self.detector.detect(
            frame_gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.marker_size_m
        )

        return detections
    
    def get_pose_apriltag_in_camera_frame(detection):
        R_ca = detection.pose_R
        t_ca = detection.pose_t
        return t_ca.flatten(), R_ca

    def draw_detections(frame, detections):
        for detection in detections:
            pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

            frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

            top_left = tuple(pts[0][0])  # First corner
            top_right = tuple(pts[1][0])  # Second corner
            bottom_right = tuple(pts[2][0])  # Third corner
            bottom_left = tuple(pts[3][0])  # Fourth corner
            cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
            cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)


class Mapper:
    """
    The Mapper watches the camera and odometry.

    It does NOT drive the robot.

    Its job:
    - Detect AprilTags
    - Classify what each tag means
    - Save where the robot was when it saw each tag
    - Estimate the tag's world position
    - Export a matplotlib map
    """

    def __init__(self, camera_matrix=None):
        if camera_matrix is None:
            # Approx camera matrix for 360p RoboMaster stream.
            # You may tune this later.
            camera_matrix = np.array([
                [314, 0, 320],
                [0, 314, 180],
                [0, 0, 1]
            ])

        self.K = camera_matrix
        self.apriltag_detector = AprilTagDetector(
            K=self.K,
            threads=2,
            marker_size_m=0.153
        )

        self.mapped_tags = {}
        self.last_tag_log_time = {}

        self.start_x = None
        self.start_y = None

        # Future YOLO obstacle memory.
        # You can add obstacles here later.
        self.obstacles = []

        # Future loading dock memory.
        # This can come from color segmentation later.
        self.loading_dock = None

    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------
    def classify_tag(self, tag_id):
        """
        Converts an AprilTag ID into a useful project object type.
        """

        if tag_id in LARGE_GOALS:
            return "large_goal"

        if tag_id in SMALL_GOALS:
            return "small_goal"

        if tag_id in CHARGING:
            return "recharge_station"

        return "unknown_tag"

    # ------------------------------------------------------------
    # Position estimate
    # ------------------------------------------------------------
    def estimate_tag_world_position(self, robot_x, robot_y, robot_yaw_deg, tag_dist, tag_bias_px):
        """
        Estimate the tag's world position from robot odometry and camera measurement.

        This is an approximation.

        Assumptions:
        - robot_x points forward in the robot's starting world frame
        - robot_y points left/right in the world frame
        - robot_yaw_deg is robot heading in degrees
        - tag_dist is distance from camera to tag
        - tag_bias_px is horizontal pixel offset from image center

        Positive tag_bias_px means the tag appears to the right of image center.
        """

        fx = self.K[0, 0]

        # Convert pixel offset to camera angle.
        # Positive right in image. We make right turn negative in world y.
        camera_angle_rad = math.atan2(tag_bias_px, fx)

        robot_yaw_rad = math.radians(robot_yaw_deg)

        # Approx heading toward tag.
        tag_heading_rad = robot_yaw_rad + camera_angle_rad

        tag_x = robot_x + tag_dist * math.cos(tag_heading_rad)
        tag_y = robot_y + tag_dist * math.sin(tag_heading_rad)

        return tag_x, tag_y

    # ------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------
    def update(self, frame, robot_x, robot_y, robot_yaw_deg):
        """
        Process one camera frame.

        This should be called once per main loop.

        Returns:
            list of tag events seen in this frame
        """

        if self.start_x is None:
            self.start_x = robot_x
            self.start_y = robot_y

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.apriltag_detector.find_tags(gray)

        events = []

        if not detections:
            return events

        for detection in detections:
            tag_id = int(detection.tag_id)

            if detection.pose_t is None:
                continue

            tag_dist = float(np.linalg.norm(detection.pose_t))

            frame_center_x = frame.shape[1] / 2.0
            tag_center_x = detection.center[0]
            tag_bias_px = float(tag_center_x - frame_center_x)

            object_type = self.classify_tag(tag_id)

            event = {
                "tag_id": tag_id,
                "type": object_type,
                "distance": tag_dist,
                "bias_px": tag_bias_px
            }

            events.append(event)

            self.ping_tag(
                tag_id=tag_id,
                object_type=object_type,
                tag_dist=tag_dist,
                tag_bias_px=tag_bias_px,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw_deg=robot_yaw_deg
            )

        return events

    def plot_robot_pos(self, x, y, *args):
        """
        For debug: plot the robot's current position on the map.
        """

        self.robot_pos.append((x, y))
        if len(self.robot_pos) > 1000:
            self.robot_pos.pop(0)



    # ------------------------------------------------------------
    # Ping / save tag
    # ------------------------------------------------------------
    def ping_tag(self, tag_id, object_type, tag_dist, tag_bias_px, robot_x, robot_y, robot_yaw_deg):
        """
        Save a tag/object location.

        We store both:
        - robot position when seen
        - estimated tag position
        """

        if tag_id in self.mapped_tags:
            return

        now = time.time()

        tag_x, tag_y = self.estimate_tag_world_position(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw_deg=robot_yaw_deg,
            tag_dist=tag_dist,
            tag_bias_px=tag_bias_px
        )

        self.mapped_tags[tag_id] = {
            "tag_id": tag_id,
            "type": object_type,

            # Estimated location of the actual tag/object.
            "x": tag_x,
            "y": tag_y,

            # Robot pose when it saw the tag.
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_yaw": robot_yaw_deg,

            "distance": tag_dist,
            "bias_px": tag_bias_px,
            "last_seen": now
        }

        self.last_tag_log_time[tag_id] = now

        print("\n[MAP] Tag pinged")
        print(f"  tag_id: {tag_id}")
        print(f"  type: {object_type}")
        print(f"  estimated tag position: x={tag_x:.3f}, y={tag_y:.3f}")
        print(f"  robot position: x={robot_x:.3f}, y={robot_y:.3f}, yaw={robot_yaw_deg:.2f}")
        print(f"  distance: {tag_dist:.3f} m")
        print(f"  bias: {tag_bias_px:.1f} px")

    # ------------------------------------------------------------
    # Future manual/object methods
    # ------------------------------------------------------------
    def add_obstacle(self, x, y, radius=0.15, label="obstacle"):
        """
        Future use for YOLO.

        When YOLO sees a fabric box, estimate its x/y and call this.
        """

        self.obstacles.append({
            "x": x,
            "y": y,
            "radius": radius,
            "label": label,
            "last_seen": time.time()
        })

        print(f"[MAP] Obstacle added at x={x:.3f}, y={y:.3f}")

    def set_loading_dock(self, x, y):
        """
        Future use for colored tape loading dock detection.
        """

        self.loading_dock = {
            "x": x,
            "y": y,
            "last_seen": time.time()
        }

        print(f"[MAP] Loading dock set at x={x:.3f}, y={y:.3f}")

    # ------------------------------------------------------------
    # Debug printing
    # ------------------------------------------------------------
    def print_map(self):
        """
        Print all currently mapped items.
        """

        print("\n========== CURRENT MAP ==========")

        if not self.mapped_tags and not self.obstacles and self.loading_dock is None:
            print("No mapped objects yet.")

        for tag_id, data in self.mapped_tags.items():
            print(
                f"Tag {tag_id}: "
                f"type={data['type']}, "
                f"tag=({data['x']:.3f}, {data['y']:.3f}), "
                f"robot_seen_from=({data['robot_x']:.3f}, {data['robot_y']:.3f}), "
                f"dist={data['distance']:.3f}"
            )

        if self.loading_dock is not None:
            print(
                f"Loading dock: "
                f"({self.loading_dock['x']:.3f}, {self.loading_dock['y']:.3f})"
            )

        for i, obstacle in enumerate(self.obstacles):
            print(
                f"Obstacle {i + 1}: "
                f"({obstacle['x']:.3f}, {obstacle['y']:.3f}), "
                f"radius={obstacle['radius']:.3f}"
            )

        print("=================================\n")

    # ------------------------------------------------------------
    # Plot map
    # ------------------------------------------------------------
    def save_map_plot(self, filename="generated_map.png"):
        """
        Save a bird's-eye-view map using matplotlib.

        Required project symbols:
        - Red circles for obstacles
        - Blue triangle for small goal
        - Green triangle for large goal
        - Yellow square for loading dock
        - Black square for recharge station
        """

        fig, ax = plt.subplots(figsize=(7, 7))

        ax.set_title("Generated Robot Map")
        ax.set_xlabel("x position [m]")
        ax.set_ylabel("y position [m]")

        ax.set_xlim(-MAP_SIZE_METERS, MAP_SIZE_METERS)
        ax.set_ylim(-MAP_SIZE_METERS, MAP_SIZE_METERS)

        ticks = np.arange(-MAP_SIZE_METERS, MAP_SIZE_METERS + GRID_SPACING, GRID_SPACING)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.grid(True, linewidth=0.4)

        # Keep equal scale so 1m x and 1m y look the same.
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        ox = self.start_x if self.start_x is not None else 0.0
        oy = self.start_y if self.start_y is not None else 0.0

        # Plot mapped tags.
        for tag_id, data in self.mapped_tags.items():
            obj_type = data["type"]
            x = data["x"] - ox
            y = data["y"] - oy

            # Clamp plot to arena bounds.
            # This prevents one bad estimate from blowing up the map view.
            if x < -1 or x > MAP_SIZE_METERS + 1 or y < -1 or y > MAP_SIZE_METERS + 1:
                continue

            if obj_type == "small_goal":
                ax.scatter(x, y, marker="^", s=180, color="blue", label="Small Goal")
                ax.text(x + 0.03, y + 0.03, f"Small {tag_id}", fontsize=8)

            elif obj_type == "large_goal":
                ax.scatter(x, y, marker="^", s=180, color="green", label="Large Goal")
                ax.text(x + 0.03, y + 0.03, f"Large {tag_id}", fontsize=8)

            elif obj_type == "recharge_station":
                ax.scatter(x, y, marker="s", s=160, color="black", label="Recharge")
                ax.text(x + 0.03, y + 0.03, f"Recharge {tag_id}", fontsize=8)

            else:
                ax.scatter(x, y, marker="x", s=120, color="purple", label="Unknown Tag")
                ax.text(x + 0.03, y + 0.03, f"Tag {tag_id}", fontsize=8)


        # Plot robot position history.
        if hasattr(self, "robot_pos") and self.robot_pos:
            rx = [pos[0] - ox for pos in self.robot_pos]
            ry = [pos[1] - oy for pos in self.robot_pos]
            ax.plot(rx, ry, color="gray", linewidth=0.2, alpha=0.3, label="Robot Path")
        
        # Plot loading dock.
        if self.loading_dock is not None:
            dx = self.loading_dock["x"] - ox
            dy = self.loading_dock["y"] - oy
            ax.scatter(
                dx,
                dy,
                marker="s",
                s=180,
                color="yellow",
                edgecolors="black",
                label="Loading Dock"
            )
            ax.text(dx + 0.03, dy + 0.03, "Loading Dock", fontsize=8)

        # Plot obstacles.
        for i, obstacle in enumerate(self.obstacles):
            dx = obstacle["x"] - ox
            dy = obstacle["y"] - oy
            circle = plt.Circle(
                (dx, dy),
                obstacle["radius"],
                color="red",
                alpha=0.7
            )
            ax.add_patch(circle)
            ax.text(dx + 0.03, dy + 0.03, f"Obstacle {i + 1}", fontsize=8)

        # Plot starting point at origin.
        if self.start_x is not None:
            ax.scatter(
                0.0,
                0.0,
                marker="*",
                s=300,
                color="orange",
                edgecolors="black",
                zorder=5,
                label="Start"
            )
            ax.text(0.03, 0.03, "Start", fontsize=8)

        # Avoid duplicate legend labels.
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper right")

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)

        print(f"[MAP] Saved map plot to {filename}")

