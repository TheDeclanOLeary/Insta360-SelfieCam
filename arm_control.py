"""xArm control for the Selfie Station.

Importing this module loads the xArm SDK but does NOT connect to the robot —
the connection is opened lazily on the first move call (get_arm), so the
FastAPI app still starts if the arm is offline.

Motion model:
- The "are we there yet?" check (`arm_at_pose`) is a *pure read* of the live
  joint angles via api_get_servo_angle. It never commands motion.
- Trajectory following (`arm_follow`) issues one blocking api_set_servo_angle
  per waypoint, so every intermediate pose is honored — it is never a single
  direct move to the goal pose.
- Homing is gated (`_arm_deployed`): `move_arm_home` only commands motion if
  `move_arm_to_selfie` armed the gate this session. A stray or automatic
  /disconnect on an arm that was never deployed is a no-op, so the arm is
  never driven into motion uncommanded.
"""

import sys
import time
import threading

# --- ARM CONFIGURATION ---
SBOT_PATH = "/home/l5vel-sbot/SBot/third_party/sbot_classical"
ARM_IP = "172.16.0.11"

# path_setup registers arm_base_control on sys.path as an import side effect.
if SBOT_PATH not in sys.path:
    sys.path.insert(0, SBOT_PATH)
import path_setup  # noqa: E402,F401
from arm_base_control.arm import XArmHandler  # noqa: E402

# Joint-angle trajectory (degrees) from home -> selfie pose.
# Index 0 is home; the last entry is the desired selfie position.
ARM_TRAJECTORY = [
    [0, 38.1, -6.7, 2, 43.3, -1.3],
    [0, 43, -30, 0, -1.5, 0],
    [0, 8, -120, 0, -56, 0],
    [-179, 8, -120, 0, -40, 0],
    [-179, 72.4, -27.8, 40.7, 24.4, -50.9],
]
ARM_HOME_POSE = ARM_TRAJECTORY[0]
ARM_SELFIE_POSE = ARM_TRAJECTORY[-1]
ARM_POSE_TOL_DEG = 2.5
ARM_SPEED = 25
ARM_MVACC = 25
ARM_SETTLE_SEC = 2  # pause after a move so the arm stabilizes before capture

_arm_handler = None
_arm_lock = threading.RLock()  # serialize arm access across requests

# Safety gate: True only after the arm has been explicitly commanded to the
# selfie pose this session. `move_arm_home` refuses to move unless this is set,
# so a stray/automatic /disconnect can never drive an undeployed arm.
_arm_deployed = False


def get_arm():
    """Lazily connect to the xArm; cached for the process lifetime."""
    global _arm_handler
    with _arm_lock:
        if _arm_handler is None:
            _arm_handler = XArmHandler(
                robot_ip=ARM_IP, gripper=None, dynamic_recovery_enabled=True
            )
    return _arm_handler


def arm_current_angles():
    """Read the live first-6 joint angles, or None if unavailable."""
    code, angles = get_arm().api_get_servo_angle(is_radian=False)
    if code != 0 or angles is None:
        return None
    return list(angles[:6])


def arm_at_pose(target, tol=ARM_POSE_TOL_DEG):
    """True only if every joint is within tol of target. Pure read — no motion."""
    angles = arm_current_angles()
    if angles is None:
        return False
    return all(abs(a - t) <= tol for a, t in zip(angles, target))


def arm_ready():
    """Clear faults and put the arm in position-control ready state."""
    arm = get_arm()
    arm.arm.clean_error()
    arm.arm.clean_warn()
    time.sleep(0.5)
    arm.arm.motion_enable(True)
    arm.api_set_mode(6)   # position control
    arm.api_set_state(0)  # ready
    time.sleep(0.2)


def arm_follow(waypoints):
    """Step through joint-angle waypoints one blocking move at a time so every
    intermediate pose is honored (never a single direct move to the goal)."""
    arm = get_arm()
    for i, wp in enumerate(waypoints):
        code = arm.api_set_servo_angle(
            angle=list(wp), speed=ARM_SPEED, mvacc=ARM_MVACC, wait=True
        )
        if code != 0:
            raise Exception(f"Arm move to waypoint {i + 1}/{len(waypoints)} failed (code {code}).")


def move_arm_to_selfie():
    """Drive the arm to the selfie pose along the forward trajectory, unless a
    fresh joint read shows it is already there.

    This is the explicit deploy command; it arms the gate (`_arm_deployed`) so
    the arm may later be sent home. The flag is set before motion so that a
    partially-completed trajectory can still be reversed by `move_arm_home`.

    Only when the arm actually travels the trajectory does it then wait
    `ARM_SETTLE_SEC` to stabilize before returning, so the capture isn't
    blurred. If it's already at the selfie pose, it returns immediately with no
    wait.
    """
    global _arm_deployed
    with _arm_lock:
        _arm_deployed = True  # commanded away from home -> homing is now allowed
        if arm_at_pose(ARM_SELFIE_POSE):
            return
        arm_ready()
        arm_follow(ARM_TRAJECTORY)
        # First arrival only: let the arm settle before the photo is taken.
        # (Skipped when already at pose via the early return above.)
        time.sleep(ARM_SETTLE_SEC)


def arm_release():
    """Reset the arm and relinquish programmatic control by switching to mode 1.

    Clears any faults/warnings first (reset), then drops the arm into mode 1 —
    the xArm servo-motion mode that gives up position-control ownership and
    allows the arm to be freely repositioned by hand or by an external
    controller. Call this only after the arm is safely parked at home.

    No-op if we never connected to the arm this session, so a disconnect with no
    prior arm activity won't open a connection just to release it.
    """
    global _arm_deployed
    with _arm_lock:
        if _arm_handler is None:
            return
        arm = _arm_handler
        arm.arm.clean_error()
        arm.arm.clean_warn()
        time.sleep(0.3)
        arm.api_set_mode(1)   # servo motion mode -> hand off control
        arm.api_set_state(0)  # apply the mode
        _arm_deployed = False


def move_arm_home():
    """Return to home along the reversed trajectory, unless already home.

    Gated: refuses to command any motion unless the arm was deployed to the
    selfie pose this session (`_arm_deployed`). This makes a stray or automatic
    /disconnect on an undeployed arm a guaranteed no-op — the arm never moves
    unless it was explicitly commanded out in the first place.
    """
    global _arm_deployed
    with _arm_lock:
        if not _arm_deployed:
            return
        if arm_at_pose(ARM_HOME_POSE):
            _arm_deployed = False
            return
        arm_ready()
        arm_follow(list(reversed(ARM_TRAJECTORY)))
        _arm_deployed = False
