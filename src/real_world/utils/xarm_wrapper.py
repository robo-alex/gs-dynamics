from xarm import version
from xarm.wrapper import XArmAPI
import numpy as np
import time
import traceback


class XARM7:
    def __init__(
        self, 
        interface="192.168.1.196", 
        init_pose=[256.7, 5.1, 400.1, 178.9, 0.0, 1.4],
        init_servo_angle=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0],
        speed=50,  # mm/s
        radian_speed=None,  # rad/s
        gripper_enable=True,
    ):
        self.pprint("xArm-Python-SDK Version:{}".format(version.__version__))
        self.alive = True
        self._arm = XArmAPI(interface, baud_checkset=False)
        self.init_pose = init_pose
        self.init_servo_angle = init_servo_angle
        self.speed = speed
        self.gripper_enable = gripper_enable
        self._robot_init(gripper_enable)

    # Robot Init
    def _robot_init(self, gripper_enable=True):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        if gripper_enable:
            self._arm.set_gripper_enable(True)
            self._arm.set_gripper_mode(0)
            self._arm.clean_gripper_error()
        # self._arm.set_collision_sensitivity(1)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, "register_count_changed_callback"):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Robot Contrl: here the pose is the end-effector pose [X, Y, Z, roll, pitch, yaw]
    def move_to_pose(self, pose, wait=True, ignore_error=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_position(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], speed=self.speed, wait=wait
        )
        if not ignore_error:
            if not self._check_code(code, "set_position"):
                raise ValueError("move_to_pose Error")
        return True

    def get_current_pose(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, pose = self._arm.get_position()
        if not self._check_code(code, "get_position"):
            import ipdb; ipdb.set_trace()
            raise ValueError("get_current_pose Error")
        return pose
    
    def get_current_joint(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, joint = self._arm.get_servo_angle()
        if not self._check_code(code, "get_servo_angle"):
            raise ValueError("get_current_joint Error")
        return joint

    def open_gripper(self, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(300, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("open_gripper Error")
        return True

    def close_gripper(self, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(-10, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("close_gripper Error")
        return True

    def get_gripper_state(self):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code, state = self._arm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            raise ValueError("get_gripper_position Error")
        return state

    def reset(self, wait=True):
        self.move_to_pose(self.init_pose, wait=wait)
        self._arm.set_servo_angle(angle=self.init_servo_angle, isradian=False, wait=wait)
        if self.gripper_enable:
            self.open_gripper(wait=wait)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._arm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            import ipdb; ipdb.set_trace()
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._arm.connected,
                    self._arm.state,
                    self._arm.error_code,
                    ret1,
                    ret2,
                )
            )
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(
                "[{}][{}] {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    stack_tuple[1],
                    " ".join(map(str, args)),
                )
            )
        except:
            print(*args, **kwargs)

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False
