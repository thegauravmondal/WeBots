"""
search_robot.py - smooth red-ball search controller for Webots.

Sensors  : front camera (red-blob detection) + 180-degree LiDAR
Actuators: two driven wheels (differential drive)

What changed versus the first version (to stop the "mad" motion)
----------------------------------------------------------------
* Wheel commands are slew-rate limited, so the robot accelerates and turns
  smoothly instead of jerking.
* The robot drives STRAIGHT at its target whenever the LiDAR shows a clear
  path; gap-following is used only when something is in the way, and it has
  hysteresis so it does not flip between two gaps.
* Side-wall repulsion is proportional (gentle) instead of an on/off kick.
* The ball must be seen in 2 consecutive frames before it is tracked, and
  tracking "coasts" for a moment if the ball flickers out of view.
* Fewer 360-degree scans (only at a few room waypoints), always in one
  direction and at a calm speed. No scan at start-up.
* A waypoint loop that passes through the doorway / pocket properly, so the
  robot no longer pushes against walls trying to reach a point behind them.

States: SEARCH -> (SCAN) / TRACK -> LOST_TURN -> GOTO_LAST -> SEARCH / FOUND
"""

import math
from controller import Supervisor

# ------------------------------------------------------------------ geometry
WHEEL_RADIUS = 0.04          # m
AXLE_LENGTH = 0.16           # m
MAX_WHEEL_SPEED = 15.0       # rad/s

# ------------------------------------------------------------------ pose
START_POSE = (1.6, -1.6, math.pi)   # must match the world file
# True  -> exact pose from the simulator (stable, recommended for the demo).
# False -> wheel odometry only (drifts when wheels slip).
USE_GROUND_TRUTH_POSE = True

# ------------------------------------------------------------------ target
BALL_DIAMETER = 0.12
STOP_DISTANCE = 0.24         # m from camera -> success
MIN_BALL_PIXELS = 8
TRACK_COAST = 0.5            # s: keep tracking after the ball flickers out

# ------------------------------------------------------------------ motion
CRUISE_SPEED = 0.45          # m/s (wheel limit is about 0.60 m/s)
TURN_GAIN = 1.6
MAX_OMEGA = 2.0              # rad/s
SPIN_SPEED = 1.8             # rad/s for scans
ACC_V = 1.2                  # m/s^2 slew limit
ACC_W = 5.0                  # rad/s^2 slew limit
LOST_TURN_TIME = 1.5
GOTO_LAST_TIMEOUT = 9.0
WAYPOINT_TIMEOUT = 18.0
WAYPOINT_RADIUS = 0.35

# ------------------------------------------------------------------ lidar
# Webots Lidar range images are ordered like a camera image (index 0 = left).
# If the robot steers the wrong way around obstacles, set this to False.
LIDAR_LEFT_FIRST = True
GAP_HALF_WIDTH = 11          # beams each side that must be clear
DIRECT_CLEAR = 0.9           # m: go straight at the target if this clear
MIN_FREE = 0.25
CLEAR_CAP = 0.9
ANGLE_WEIGHT = 0.6
HYSTERESIS = 0.12            # keep the previous gap unless clearly worse
SIDE_DANGER = 0.17           # m

# (x, y, scan_here) - loop that covers every room, the doorway and the pocket
WAYPOINTS = [
    (0.6, -1.6, False),
    (-1.0, -1.8, False),     # narrow gap under the left barrier
    (-1.5, -1.4, True),
    (-1.5, -0.2, False),
    (-0.6, 0.5, False),
    (0.0, 0.6, False),
    (0.0, 0.9, False),       # doorway
    (0.0, 1.5, True),
    (-0.6, 1.17, False),
    (-1.0, 1.17, False),     # pocket entrance
    (-1.5, 1.4, True),
    (-1.0, 1.17, False),
    (-0.6, 1.17, False),
    (1.3, 1.5, True),
    (0.0, 1.25, False),
    (0.0, 0.9, False),       # doorway again
    (0.65, 0.6, False),
    (0.65, -0.1, False),
    (0.0, -0.75, True),
    (0.65, -0.1, False),
    (1.5, 0.5, False),
    (1.5, -0.6, True),
    (1.5, -1.7, False),
]

SEARCH, SCAN, TRACK, LOST_TURN, GOTO_LAST, FOUND = (
    "SEARCH", "SCAN", "TRACK", "LOST_TURN", "GOTO_LAST", "FOUND")


def wrap(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class SearchRobot:
    def __init__(self):
        self.robot = Supervisor()
        self.dt = int(self.robot.getBasicTimeStep())
        self.dt_s = self.dt / 1000.0

        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        self.left_enc = self.robot.getDevice("left wheel sensor")
        self.right_enc = self.robot.getDevice("right wheel sensor")
        self.left_enc.enable(self.dt)
        self.right_enc.enable(self.dt)

        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.dt)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()
        self.focal_px = (self.cam_w / 2.0) / math.tan(self.camera.getFov() / 2.0)

        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(self.dt)
        self.n = self.lidar.getHorizontalResolution()
        self.l_fov = self.lidar.getFov()
        self.l_max = self.lidar.getMaxRange()
        self.l_step = self.l_fov / (self.n - 1)
        if LIDAR_LEFT_FIRST:
            self.angles = [self.l_fov / 2.0 - i * self.l_step for i in range(self.n)]
        else:
            self.angles = [-self.l_fov / 2.0 + i * self.l_step for i in range(self.n)]
        self.front_idx = [i for i, a in enumerate(self.angles) if abs(a) < math.radians(15)]
        self.left_idx = [i for i, a in enumerate(self.angles)
                         if math.radians(50) <= a <= math.radians(90)]
        self.right_idx = [i for i, a in enumerate(self.angles)
                          if -math.radians(90) <= a <= -math.radians(50)]
        self.ranges = [self.l_max] * self.n
        self.front = self.left_side = self.right_side = self.l_max

        self.node = self.robot.getSelf()
        self.x, self.y, self.th = START_POSE
        self.dth = 0.0
        self.prev_enc = None

        self.cmd_v = 0.0
        self.cmd_w = 0.0

        self.t = 0.0
        self.state = SEARCH
        self.state_t0 = 0.0
        self.wp_idx = 0
        self.wp_t0 = 0.0
        self.scan_acc = 0.0
        self.steer = 0.0
        self.gap_angle = 0.0
        self.blocked_time = 0.0
        self.escape_back_until = 0.0
        self.escape_until = 0.0
        self.escape_dir = 1

        self.seen_frames = 0
        self.last_angle = 0.0
        self.last_dist = 1.0
        self.last_ball_t = -99.0
        self.ball_xy = None
        self.lost_since = 0.0

    # ------------------------------------------------------------ actuation
    def set_velocity(self, v, w):
        """Slew-limited command. v in m/s, w in rad/s (CCW positive)."""
        self.cmd_v += clamp(v - self.cmd_v, -ACC_V * self.dt_s, ACC_V * self.dt_s)
        self.cmd_w += clamp(w - self.cmd_w, -ACC_W * self.dt_s, ACC_W * self.dt_s)
        vl = (self.cmd_v - self.cmd_w * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        vr = (self.cmd_v + self.cmd_w * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        m = max(abs(vl), abs(vr))
        if m > MAX_WHEEL_SPEED:
            vl *= MAX_WHEEL_SPEED / m
            vr *= MAX_WHEEL_SPEED / m
        self.left_motor.setVelocity(vl)
        self.right_motor.setVelocity(vr)

    # --------------------------------------------------------------- sensing
    def update_pose(self):
        old_th = self.th
        if USE_GROUND_TRUTH_POSE and self.node is not None:
            p = self.node.getPosition()
            r = self.node.getOrientation()
            self.x, self.y = p[0], p[1]
            self.th = math.atan2(r[3], r[0])
        else:
            l = self.left_enc.getValue()
            r = self.right_enc.getValue()
            if math.isnan(l) or math.isnan(r):
                return
            if self.prev_enc is None:
                self.prev_enc = (l, r)
                return
            dl = (l - self.prev_enc[0]) * WHEEL_RADIUS
            dr = (r - self.prev_enc[1]) * WHEEL_RADIUS
            self.prev_enc = (l, r)
            ds = 0.5 * (dl + dr)
            dth = (dr - dl) / AXLE_LENGTH
            self.x += ds * math.cos(self.th + dth / 2.0)
            self.y += ds * math.sin(self.th + dth / 2.0)
            self.th = wrap(self.th + dth)
        self.dth = wrap(self.th - old_th)

    def read_lidar(self):
        raw = self.lidar.getRangeImage()
        if not raw:
            return
        inf = float("inf")
        self.ranges = [min(r, self.l_max) if (r == r and r != inf) else self.l_max
                       for r in raw]
        self.front = min(self.ranges[i] for i in self.front_idx)
        self.left_side = min(self.ranges[i] for i in self.left_idx)
        self.right_side = min(self.ranges[i] for i in self.right_idx)

    def detect_ball(self):
        """(angle, distance) of the red blob or None. angle: CCW positive."""
        img = self.camera.getImage()
        if not img:
            return None
        w, h = self.cam_w, self.cam_h
        count = 0
        sum_x = 0
        min_x, max_x = w, -1
        for y in range(h):
            row = y * w * 4
            for x in range(w):
                p = row + 4 * x
                r = img[p + 2]
                if r > 80:
                    g = img[p + 1]
                    b = img[p]
                    if r > 2.5 * g and r > 2.5 * b:
                        count += 1
                        sum_x += x
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
        if count < MIN_BALL_PIXELS:
            return None
        cx = sum_x / count
        angle = -math.atan((cx - (w - 1) / 2.0) / self.focal_px)
        width_px = max(1, max_x - min_x + 1)
        return angle, BALL_DIAMETER * self.focal_px / width_px

    def remember(self, angle, dist):
        self.last_angle = angle
        self.last_dist = dist
        cam_x = self.x + 0.10 * math.cos(self.th)
        cam_y = self.y + 0.10 * math.sin(self.th)
        self.ball_xy = (cam_x + dist * math.cos(self.th + angle),
                        cam_y + dist * math.sin(self.th + angle))

    # ------------------------------------------------------------ navigation
    def index_of(self, angle):
        if LIDAR_LEFT_FIRST:
            i = round((self.l_fov / 2.0 - angle) / self.l_step)
        else:
            i = round((angle + self.l_fov / 2.0) / self.l_step)
        return int(clamp(i, 0, self.n - 1))

    def window_free(self, i):
        lo = max(0, i - GAP_HALF_WIDTH)
        hi = min(self.n - 1, i + GAP_HALF_WIDTH)
        return min(self.ranges[lo:hi + 1])

    def choose_heading(self, desired):
        """Go straight if clear; otherwise follow the best gap (with hysteresis)."""
        if abs(desired) <= self.l_fov / 2.0 - 0.05:
            if self.window_free(self.index_of(desired)) >= DIRECT_CLEAR:
                self.gap_angle = desired
                return desired

        best_a, best_score = None, -1e9
        for i in range(self.n):
            free = self.window_free(i)
            if free < MIN_FREE:
                continue
            a = self.angles[i]
            score = min(free, CLEAR_CAP) - ANGLE_WEIGHT * abs(wrap(a - desired))
            if score > best_score:
                best_score, best_a = score, a
        if best_a is None:
            return None

        ip = self.index_of(self.gap_angle)
        fp = self.window_free(ip)
        if fp >= MIN_FREE:
            sp = min(fp, CLEAR_CAP) - ANGLE_WEIGHT * abs(wrap(self.angles[ip] - desired))
            if sp >= best_score - HYSTERESIS:
                best_a = self.angles[ip]
        self.gap_angle = best_a
        return best_a

    def open_side(self):
        half = self.n // 2
        first, second = sum(self.ranges[:half]), sum(self.ranges[half:])
        left, right = (first, second) if LIDAR_LEFT_FIRST else (second, first)
        return 1 if left >= right else -1

    def drive_toward(self, desired, max_speed=CRUISE_SPEED):
        a = self.choose_heading(desired)
        if a is None:                         # boxed in: turn toward open side
            return 0.0, self.open_side() * 1.2
        self.steer = 0.5 * self.steer + 0.5 * a
        a = self.steer
        w = clamp(TURN_GAIN * a, -MAX_OMEGA, MAX_OMEGA)
        v = max_speed * clamp(1.0 - abs(a) / 1.3, 0.2, 1.0)
        v *= clamp((self.front - 0.15) / 0.5, 0.0, 1.0)
        if abs(desired) > 1.9:                # target is behind: pivot first
            v = 0.0
        # gentle, proportional wall repulsion
        if self.left_side < SIDE_DANGER:
            w -= 6.0 * (SIDE_DANGER - self.left_side)
        if self.right_side < SIDE_DANGER:
            w += 6.0 * (SIDE_DANGER - self.right_side)
        if self.front < 0.2 and v < 0.03:     # avoid dead-lock facing a wall
            w = math.copysign(max(abs(w), 0.9), a if abs(a) > 1e-3 else self.open_side())
        return v, w

    def desired_to(self, tx, ty):
        return wrap(math.atan2(ty - self.y, tx - self.x) - self.th)

    # -------------------------------------------------------------- behaviours
    def begin_scan(self):
        self.state = SCAN
        self.scan_acc = 0.0

    def next_waypoint(self):
        self.wp_idx = (self.wp_idx + 1) % len(WAYPOINTS)
        self.wp_t0 = self.t

    def do_search(self):
        tx, ty, scan_here = WAYPOINTS[self.wp_idx]
        if math.hypot(tx - self.x, ty - self.y) < WAYPOINT_RADIUS:
            self.next_waypoint()
            if scan_here:
                self.begin_scan()
                return 0.0, 0.0
        elif self.t - self.wp_t0 > WAYPOINT_TIMEOUT:
            self.next_waypoint()
        tx, ty, _ = WAYPOINTS[self.wp_idx]
        return self.drive_toward(self.desired_to(tx, ty))

    def do_scan(self):
        self.scan_acc += abs(self.dth)
        if self.scan_acc >= 2.0 * math.pi:
            self.state = SEARCH
            self.wp_t0 = self.t
            return 0.0, 0.0
        return 0.0, SPIN_SPEED

    def do_track(self, angle, dist):
        v_max = clamp(0.5 * (dist - STOP_DISTANCE) + 0.10, 0.10, CRUISE_SPEED)
        return self.drive_toward(angle, v_max)

    def do_lost_turn(self):
        if self.t - self.lost_since < LOST_TURN_TIME:
            return 0.0, math.copysign(1.0, self.last_angle if self.last_angle else 1.0)
        if self.ball_xy is not None:
            self.state = GOTO_LAST
            self.state_t0 = self.t
        else:
            self.state = SEARCH
            self.wp_t0 = self.t
        return 0.0, 0.0

    def do_goto_last(self):
        bx, by = self.ball_xy
        if (math.hypot(bx - self.x, by - self.y) < 0.4
                or self.t - self.state_t0 > GOTO_LAST_TIMEOUT):
            self.ball_xy = None
            self.begin_scan()
            return 0.0, 0.0
        return self.drive_toward(self.desired_to(bx, by))

    # ------------------------------------------------------------------ escape
    def check_stuck(self):
        if self.state in (SEARCH, TRACK, GOTO_LAST) and self.front < 0.12:
            self.blocked_time += self.dt_s
        else:
            self.blocked_time = 0.0
        if self.blocked_time > 0.8:
            self.blocked_time = 0.0
            self.escape_back_until = self.t + 0.6
            self.escape_until = self.t + 1.6
            self.escape_dir = self.open_side()

    def do_escape(self):
        if self.t < self.escape_back_until:
            return -0.15, 0.0
        return 0.0, self.escape_dir * 1.5

    # -------------------------------------------------------------------- loop
    def decide(self, ball):
        if self.state == FOUND:
            return 0.0, 0.0

        confirmed = False
        if ball is not None:
            self.seen_frames += 1
            confirmed = self.seen_frames >= 2 or self.state == TRACK
        else:
            self.seen_frames = 0

        if confirmed:
            angle, dist = ball
            self.remember(angle, dist)
            self.last_ball_t = self.t
            if dist <= STOP_DISTANCE:
                self.state = FOUND
                print("[search_robot] Red ball found at t=%.1f s "
                      "(distance %.2f m, est. position x=%.2f y=%.2f)"
                      % (self.t, dist, self.ball_xy[0], self.ball_xy[1]))
                return 0.0, 0.0
            if self.state != TRACK:
                print("[search_robot] Ball sighted - tracking.")
            self.state = TRACK
            return self.do_track(angle, dist)

        if self.state == TRACK:
            if self.t - self.last_ball_t < TRACK_COAST:
                return self.do_track(self.last_angle, self.last_dist)
            self.state = LOST_TURN
            self.lost_since = self.t

        if self.t < self.escape_until:
            return self.do_escape()
        self.check_stuck()

        if self.state == LOST_TURN:
            return self.do_lost_turn()
        if self.state == GOTO_LAST:
            return self.do_goto_last()
        if self.state == SCAN:
            return self.do_scan()
        return self.do_search()

    def run(self):
        while self.robot.step(self.dt) != -1:
            self.t = self.robot.getTime()
            self.update_pose()
            self.read_lidar()
            v, w = self.decide(self.detect_ball())
            self.set_velocity(v, w)


if __name__ == "__main__":
    SearchRobot().run()
