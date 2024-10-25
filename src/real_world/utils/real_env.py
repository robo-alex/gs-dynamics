from typing import Optional
import sys
sys.path.append('.')
import os

import cv2
import json
import time
import pickle
import numpy as np
import torch
import math

from multiprocessing.managers import SharedMemoryManager
from real_world.camera.multi_realsense import MultiRealsense, SingleRealsense

from real_world.utils.xarm_wrapper import XARM7
from real_world.utils.pcd_utils import depth2fgpcd, rpy_to_rotation_matrix


class RealEnv:
    def __init__(self, 
            use_camera=True,
            WH=[640, 480],
            capture_fps=15,
            obs_fps=15,
            n_obs_steps=2,
            enable_color=True,
            enable_depth=True,
            process_depth=False,
            use_robot=True,
            verbose=False,
            gripper_enable=False,
            speed=50,
            push_length=0.01,
            wrist=None,
        ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_camera = use_camera
        
        if self.use_camera:
            self.WH = WH
            self.capture_fps = capture_fps
            self.obs_fps = obs_fps
            self.n_obs_steps = n_obs_steps
            if wrist is None:
                wrist = '246322303954'  # default wrist camera serial number
            self.calibrate_result_dir = 'output/latest_calibration'
            os.makedirs(self.calibrate_result_dir, exist_ok=True)
            self.vis_dir = f'{self.calibrate_result_dir}/vis'
            os.makedirs(self.vis_dir, exist_ok=True)

            self.serial_numbers = SingleRealsense.get_connected_devices_serial()
            if wrist in self.serial_numbers:
                print('Found wrist camera.')
                self.serial_numbers.remove(wrist)
                self.serial_numbers = self.serial_numbers + [wrist]  # put the wrist camera at the end
                self.n_fixed_cameras = len(self.serial_numbers) - 1
            else:
                self.n_fixed_cameras = len(self.serial_numbers)
            print(f'Found {self.n_fixed_cameras} fixed cameras.')

            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
            self.realsense =  MultiRealsense(
                    serial_numbers=self.serial_numbers,
                    shm_manager=self.shm_manager,
                    resolution=(self.WH[0], self.WH[1]),
                    capture_fps=self.capture_fps,
                    enable_color=enable_color,
                    enable_depth=enable_depth,
                    process_depth=process_depth,
                    verbose=verbose)
            self.realsense.set_exposure(exposure=100, gain=60)
            self.realsense.set_white_balance(3800)
            self.last_realsense_data = None
            self.enable_color = enable_color
            self.enable_depth = enable_depth

            self.calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            self.calibration_board = cv2.aruco.CharucoBoard(
                size=(6, 5),
                squareLength=0.04,
                markerLength=0.03,
                dictionary=self.calibration_dictionary,
            )
            self.calibration_parameters = cv2.aruco.DetectorParameters()
            calibration_parameters =  cv2.aruco.CharucoParameters()
            self.charuco_detector = cv2.aruco.CharucoDetector(
                self.calibration_board,
                calibration_parameters,
            )
            self.R_cam2world = None
            self.t_cam2world = None
            self.R_base2world = None
            self.t_base2world = None

        self.use_robot = use_robot
        self.push_length = push_length
        if self.use_robot:
            self.robot = XARM7(gripper_enable=gripper_enable, speed=speed)
            self.gripper_enable = gripper_enable

        self.bbox = np.array([[0.0, 0.6], [-0.35, 0.45], [-0.10, 0.05]])  # the world frame robot workspace
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame
        self.world_y = 0.01  # the world y coordinate of the eef during action
        self.state = None

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return (self.realsense.is_ready if self.use_camera else True) and (self.robot.is_alive if self.use_robot else True)
    
    def start(self, wait=True, exposure_time=5):
        self.realsense.start(wait=False, put_start_time=time.time() + exposure_time)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
    
    def stop_wait(self):
        self.realsense.stop_wait()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        robot_obs = dict()
        if self.use_robot:
            robot_obs['joint_angles'] = self.robot.get_current_joint()
            robot_obs['pose'] = self.robot.get_current_pose()
            if self.gripper_enable:
                robot_obs['gripper_position'] = self.robot.get_gripper_state()

        # align camera obs timestamps
        dt = 1 / self.obs_fps
        timestamp_list = [x['timestamp'][-1] for x in self.last_realsense_data.values()]
        last_timestamp = np.max(timestamp_list)
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)
        # the last timestamp is the latest one

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            if get_color:
                assert self.enable_color
                camera_obs[f'color_{camera_idx}'] = value['color'][this_idxs]  # BGR
            if get_depth and isinstance(camera_idx, int):
                assert self.enable_depth
                camera_obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def start_recording(self, file_path, start_time=None):
        self.realsense.start_recording(file_path, start_time=start_time)

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return (
            [self.R_cam2world[i].copy() for i in self.serial_numbers[:4]],
            [self.t_cam2world[i].copy() for i in self.serial_numbers[:4]],
        )

    def get_bbox(self):
        return self.bbox.copy()
    
    def decode_action(self, action):
        x_start = action[0]
        z_start = action[1]
        theta = action[2]
        action_repeat = int(action[3])
        x_end = x_start - self.push_length * action_repeat * np.cos(theta)
        z_end = z_start - self.push_length * action_repeat * np.sin(theta)
        return x_start, z_start, x_end, z_end

    def step(self, action, decoded=False):
        assert self.use_robot
        assert self.is_ready
        eef_cur = self.get_eef_points()[0]

        if decoded:
            x_start, z_start, x_end, z_end = action
        else:
            x_start, z_start, x_end, z_end = self.decode_action(action)

        yaw = None
        assert -eef_cur[2] > 0

        self.reset_robot()
        # time.sleep(0.5)
        self.move_to_table_position([x_start, self.world_y + 0.10, z_start], yaw, wait=True)
        self.move_to_table_position([x_start, self.world_y, z_start], yaw, wait=True)
        # time.sleep(0.5)
        self.move_to_table_position([x_end, self.world_y, z_end], yaw, wait=True)
        self.move_to_table_position([x_end, self.world_y + 0.10, z_end], yaw, wait=True)
        # time.sleep(0.5)
        self.reset_robot()
    
    def step_gripper(self, action, decoded=False):
        assert self.use_robot
        assert self.gripper_enable
        assert self.is_ready
        if decoded:
            x_start, z_start, x_end, z_end = action
        else:
            x_start, z_start, x_end, z_end = self.decode_action(action)

        yaw = 180 - np.arctan2(z_end - z_start, x_end - x_start) / np.pi * 180

        x_start = x_start - 0.005 * (x_end - x_start) / np.sqrt((x_end - x_start) ** 2 + (z_end - z_start) ** 2)
        z_start = z_start - 0.005 * (z_end - z_start) / np.sqrt((x_end - x_start) ** 2 + (z_end - z_start) ** 2)

        self.reset_robot()
        self.move_to_table_position([x_start, self.world_y + 0.10, z_start], yaw, wait=True)
        self.move_to_table_position([x_start, self.world_y, z_start], yaw, wait=True)
        time.sleep(5)
        self.robot.close_gripper()
        time.sleep(0.5)
        self.move_to_table_position([x_start, self.world_y + 0.02, z_start], yaw, wait=True)
        self.move_to_table_position([x_end, self.world_y + 0.02, z_end], yaw, wait=True)
        self.robot.open_gripper()
        time.sleep(0.5)
        self.move_to_table_position([x_end, self.world_y + 0.10, z_end], yaw, wait=True)
        self.reset_robot()

    def move_to_table_position(self, position, yaw=None, wait=True):
        assert self.use_robot
        assert self.is_ready
        if yaw:
            # perpendicular to the pushing direction
            rpy = np.array([180., 0., yaw])
        else:
            rpy = np.array([180., 0., 0.])
        R_gripper2base = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])

        # (x, -z, y) to (x, y, z)
        position = np.array(position)
        position = np.array([position[0], position[2], -position[1]])

        R_base2world = self.R_base2world
        t_base2world = self.t_base2world

        R_world2base = R_base2world.T
        t_world2base = -R_base2world.T @ t_base2world

        finger_in_world = position
        finger_in_base = R_world2base @ finger_in_world + t_world2base
        gripper_in_base = finger_in_base - R_gripper2base @ self.eef_point[0]

        pose = np.concatenate([gripper_in_base * 1000, rpy], axis=0)
        self.robot.move_to_pose(pose=pose, wait=wait, ignore_error=True)

    def get_robot_pose(self, raw=False):
        raw_pose = self.robot.get_current_pose()
        if raw:
            return raw_pose
        else:
            R_gripper2base = rpy_to_rotation_matrix(
                raw_pose[3], raw_pose[4], raw_pose[5]
            )
            t_gripper2base = np.array(raw_pose[:3]) / 1000
        return R_gripper2base, t_gripper2base

    def set_robot_pose(self, pose, wait=True):
        self.robot.move_to_pose(pose=pose, wait=wait, ignore_error=True)
    
    def reset_robot(self, wait=True):
        self.robot.reset(wait=wait)
    
    def hand_eye_calibrate(self, visualize=True, save=True, return_results=True):
        self.reset_robot()
        time.sleep(1)

        poses = [
            [522.6,-1.6,279.5,179.2,0,0.3],
            [494.3,133,279.5,179.2,0,-24.3],
            [498.8,-127.3,314.9,179.3,0,31.1],
            [589.5,16.6,292.9,-175,17,1.2],
            [515.8,178.5,469.2,-164.3,17.5,-90.8],
            [507.9,-255.5,248.5,-174.6,-16.5,50.3],
            [507.9,258.2,248.5,-173.5,-8,-46.8],
            [569,-155.6,245.8,179.5,3.7,49.7],
            [570.8,-1.2,435,-178.5,52.3,-153.9],
            [474.3,12.5,165.3,179.3,-15,0.3],
        ]
        R_gripper2base = []
        t_gripper2base = []
        R_board2cam = []
        t_board2cam = []
        
        for pose in poses:
            # Move to the pose and wait for 5s to make it stable
            self.set_robot_pose(pose)
            time.sleep(5)

            # Calculate the markers
            obs = self.get_obs()

            pose_real = obs['pose']
            calibration_img = obs[f'color_{self.n_fixed_cameras}'][-1]

            intr = self.get_intrinsics()[-1]
            dist_coef = np.zeros(5)

            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_handeye_img_{pose}.jpg', calibration_img)

            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            # calibrate
            corners, ids, rejected_img_points = self.aruco_detector.detectMarkers(calibration_img)
            detected_corners, detected_ids, rejected_corners, recovered_ids = self.aruco_detector.refineDetectedMarkers(
                detectedCorners=corners, 
                detectedIds=ids,
                rejectedCorners=rejected_img_points,
                image=calibration_img,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
            )

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), detected_corners, detected_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_marker_handeye_{pose}.jpg', calibration_img_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners=detected_corners,
                ids=detected_ids,
                board=self.calibration_board,
                cameraMatrix=intr,
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef ,rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_handeye_{pose}.jpg", calibration_img_vis)

            if not retval:
                raise ValueError("pose estimation failed")

            # Save the transformation of board2cam
            R_board2cam.append(cv2.Rodrigues(rvec)[0])
            t_board2cam.append(tvec[:, 0])

            # Save the transformation of the gripper2base
            print("Current pose: ", pose_real)

            R_gripper2base.append(
                rpy_to_rotation_matrix(
                    pose_real[3], pose_real[4], pose_real[5]
                )
            )
            t_gripper2base.append(np.array(pose_real[:3]) / 1000)
        
        self.reset_robot()

        R_base2gripper = []
        t_base2gripper = []
        for i in range(len(R_gripper2base)):
            R_base2gripper.append(R_gripper2base[i].T)
            t_base2gripper.append(-R_gripper2base[i].T @ t_gripper2base[i])

        # Do the robot-world hand-eye calibration
        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
            R_world2cam=R_board2cam,
            t_world2cam=t_board2cam,
            R_base2gripper=R_base2gripper,
            t_base2gripper=t_base2gripper,
            R_base2world=None,
            t_base2world=None,
            R_gripper2cam=None,
            t_gripper2cam=None,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        t_gripper2cam = t_gripper2cam[:, 0]  # (3, 1) -> (3,)
        t_base2world = t_base2world[:, 0]  # (3, 1) -> (3,)

        results = {}
        results["R_gripper2cam"] = R_gripper2cam
        results["t_gripper2cam"] = t_gripper2cam
        results["R_base2world"] = R_base2world
        results["t_base2world"] = t_base2world

        print("R_gripper2cam", R_gripper2cam)
        print("t_gripper2cam", t_gripper2cam)
        if save:
            with open(f"{self.calibrate_result_dir}/calibration_handeye_result.pkl", "wb") as f:
                pickle.dump(results, f)
        if return_results:
            return results

    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        rvecs = {}
        tvecs = {}
        rvecs_list = []
        tvecs_list = []

        # Calculate the markers
        obs = self.get_obs(get_depth=visualize)
        intrs = self.get_intrinsics()
        dist_coef = np.zeros(5)

        for i in range(self.n_fixed_cameras):  # ignore the wrist camera
            device = self.serial_numbers[i]
            intr = intrs[i]
            calibration_img = obs[f'color_{i}'][-1].copy()
            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_img_{device}.jpg', calibration_img)
            
            calibration_img = cv2.cvtColor(calibration_img, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(calibration_img)

            if visualize:
                calibration_img_vis = cv2.aruco.drawDetectedMarkers(calibration_img.copy(), marker_corners, marker_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{device}.jpg', calibration_img_vis)

                calibration_depth = obs[f'depth_{i}'][-1].copy()
                calibration_depth = np.minimum(calibration_depth, 2.0)
                calibration_depth_vis = calibration_depth / calibration_depth.max() * 255
                calibration_depth_vis = calibration_depth_vis[:, :, np.newaxis].repeat(3, axis=2)
                calibration_depth_vis = cv2.applyColorMap(calibration_depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(f'{self.vis_dir}/calibration_depth_{device}.jpg', calibration_depth_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, 
                charuco_ids, 
                self.calibration_board, 
                cameraMatrix=intr, 
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )

            if not retval:
                print("pose estimation failed")
                import ipdb; ipdb.set_trace()

            if visualize:
                calibration_img_vis = calibration_img.copy()[:, :, np.newaxis].repeat(3, axis=2)
                cv2.drawFrameAxes(calibration_img_vis, intr, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{device}.jpg", calibration_img_vis)

            rvecs[device] = rvec
            tvecs[device] = tvec
            rvecs_list.append(rvec)
            tvecs_list.append(tvec)
        
        if save:
            # save rvecs, tvecs
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'wb') as f:
                pickle.dump(rvecs, f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'wb') as f:
                pickle.dump(tvecs, f)
            
            # save rvecs, tvecs, intrinsics as numpy array
            rvecs_list = np.array(rvecs_list)
            tvecs_list = np.array(tvecs_list)
            intrs = np.array(intrs)
            with open(f'{self.calibrate_result_dir}/rvecs.npy', 'wb') as f:
                np.save(f, rvecs_list)
            with open(f'{self.calibrate_result_dir}/tvecs.npy', 'wb') as f:
                np.save(f, tvecs_list)
            with open(f'{self.calibrate_result_dir}/intrinsics.npy', 'wb') as f:
                np.save(f, intrs)

        if return_results:
            return rvecs, tvecs

    def calibrate(self, re_calibrate=False, visualize=True):
        if re_calibrate:
            if self.use_robot:
                R_base2board = np.array([
                    [1.0, 0, 0],
                    [0, -1.0, 0],
                    [0, 0, -1.0]
                ])
                t_base2board = np.array(
                    [-0.095, 0.085, -0.01]
                )
                with open(f'{self.calibrate_result_dir}/base.pkl', 'wb') as f:
                    pickle.dump({'R_base2world': R_base2board, 't_base2world': t_base2board}, f)
            else:
                if os.path.exists(f'{self.calibrate_result_dir}/base.pkl'):
                    with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
                        base = pickle.load(f)
                    R_base2board = base['R_base2world']
                    t_base2board = base['t_base2world']
                else:
                    R_base2board = None
                    t_base2board = None
            rvecs, tvecs = self.fixed_camera_calibrate(visualize=visualize)
            print('calibration finished')
        else:
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
                rvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
                tvecs = pickle.load(f)
            with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
                base = pickle.load(f)
            R_base2board = base['R_base2world']
            t_base2board = base['t_base2world']

        self.R_cam2world = {}
        self.t_cam2world = {}
        self.R_base2world = R_base2board
        self.t_base2world = t_base2board

        for i in range(self.n_fixed_cameras):
            device = self.serial_numbers[i]
            R_world2cam = cv2.Rodrigues(rvecs[device])[0]
            t_world2cam = tvecs[device][:, 0]
            self.R_cam2world[device] = R_world2cam.T
            self.t_cam2world[device] = -R_world2cam.T @ t_world2cam
        
        if visualize:
            self.verify_eef_points()
    
    def verify_eef_points(self):
        if self.last_realsense_data is None:
            _ = self.get_obs()
        eef_pos = self.get_eef_points()
        extr = self.get_extrinsics()
        intr = self.get_intrinsics()
        for i in range(self.n_fixed_cameras):
            device = self.serial_numbers[i]
            R_cam2world = extr[0][i]
            t_cam2world = extr[1][i]
            R_world2cam = R_cam2world.T
            t_world2cam = -R_cam2world.T @ t_cam2world
            eef_pos_in_cam = R_world2cam @ eef_pos.T + t_world2cam[:, np.newaxis]
            eef_pos_in_cam = eef_pos_in_cam.T
            fx, fy, cx, cy = intr[i][0, 0], intr[i][1, 1], intr[i][0, 2], intr[i][1, 2]
            eef_pos_in_cam = eef_pos_in_cam / eef_pos_in_cam[:, 2][:, np.newaxis]
            eef_pos_in_cam = eef_pos_in_cam[:, :2]
            eef_pos_in_cam[:, 0] = eef_pos_in_cam[:, 0] * fx + cx
            eef_pos_in_cam[:, 1] = eef_pos_in_cam[:, 1] * fy + cy
            eef_pos_in_cam = eef_pos_in_cam.astype(int)
            color_img = self.last_realsense_data[i]['color'][-1].copy()
            for pos in eef_pos_in_cam:
                cv2.circle(color_img, tuple(pos), 5, (255, 0, 0), -1)
            eef_pos_axis = np.array([[[0, 0, 0], [0, 0, 0.1]], [[0, 0, 0], [0, 0.1, 0]], [[0, 0, 0], [0.1, 0, 0]]])
            axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for j, axis in enumerate(eef_pos_axis):
                axis_eef = axis + eef_pos[0]
                axis_in_cam = R_world2cam @ axis_eef.T + t_world2cam[:, np.newaxis]
                axis_in_cam = axis_in_cam.T
                axis_in_cam = axis_in_cam / axis_in_cam[:, 2][:, np.newaxis]
                axis_in_cam = axis_in_cam[:, :2]
                axis_in_cam[:, 0] = axis_in_cam[:, 0] * fx + cx
                axis_in_cam[:, 1] = axis_in_cam[:, 1] * fy + cy
                axis_in_cam = axis_in_cam.astype(int)
                cv2.line(color_img, tuple(axis_in_cam[0]), tuple(axis_in_cam[1]), axis_colors[j], 2)
            cv2.imwrite(f'{self.vis_dir}/eef_in_cam_{device}.jpg', color_img)
            
        vis_3d = True
        if vis_3d:
            import open3d as o3d
            points_list = []
            colors_list = []
            for i in range(self.n_fixed_cameras):
                device = self.serial_numbers[i]
                color_img = self.last_realsense_data[i]['color'][-1].copy()
                depth_img = self.last_realsense_data[i]['depth'][-1].copy() / 1000.0
                intr = self.get_intrinsics()[i]
                extr = self.get_extrinsics()
                mask = np.logical_and(depth_img > 0, depth_img < 2.0).reshape(-1)
                mask = mask[:, None].repeat(3, axis=1)
                points = depth2fgpcd(depth_img, intr).reshape(-1, 3)
                colors = color_img.reshape(-1, 3)[:, ::-1]
                points = points[mask].reshape(-1, 3)
                colors = colors[mask].reshape(-1, 3)
                R_cam2world = extr[0][i]
                t_cam2world = extr[1][i]
                points = R_cam2world @ points.T + t_cam2world[:, np.newaxis]
                points_list.append(points.T)
                colors_list.append(colors)

            points = np.concatenate(points_list, axis=0)
            colors = np.concatenate(colors_list, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)
            
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=self.bbox[:, 0], max_bound=self.bbox[:, 1]))
            o3d.visualization.draw_geometries([pcd])
            pcd = pcd.voxel_down_sample(voxel_size=0.001)
            outliers = None
            new_outlier = None
            rm_iter = 0
            while new_outlier is None or len(new_outlier.points) > 0:
                _, inlier_idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0 + rm_iter * 0.5)
                new_pcd = pcd.select_by_index(inlier_idx)
                new_outlier = pcd.select_by_index(inlier_idx, invert=True)
                if outliers is None:
                    outliers = new_outlier
                else:
                    outliers += new_outlier
                pcd = new_pcd
                rm_iter += 1

            pcd_eef = o3d.geometry.PointCloud()
            pcd_eef.points = o3d.utility.Vector3dVector(eef_pos)
            pcd_eef.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0]]))
            o3d.visualization.draw_geometries([pcd, pcd_eef])

    def get_eef_points(self):
        assert self.R_base2world is not None
        assert self.t_base2world is not None
        R_gripper2base, t_gripper2base = self.get_robot_pose()
        R_gripper2world = self.R_base2world @ R_gripper2base
        t_gripper2world = self.R_base2world @ t_gripper2base + self.t_base2world
        stick_point_in_world = R_gripper2world @ self.eef_point.T + t_gripper2world[:, np.newaxis]
        stick_point_in_world = stick_point_in_world.T
        return stick_point_in_world

