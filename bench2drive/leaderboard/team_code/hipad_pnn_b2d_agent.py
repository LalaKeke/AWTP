import os
import sys
import cv2
import json
import time
import math
import copy
import carla
import torch
import pathlib
import datetime
import importlib
import numpy as np

from PIL import Image
from scipy.optimize import fsolve
from pyquaternion import Quaternion
from torchvision import transforms as T

from mmcv import Config
from mmcv.runner import (
    load_checkpoint,
    wrap_fp16_model,
)
from mmcv.parallel import DataContainer
from mmcv.parallel.collate import collate as mm_collate_to_batch_form

from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose

from team_code.planner import RoutePlanner
from team_code.pid_controller import PIDController
from team_code.pnn_horizon_controller import PNNHorizonController
from team_code.pnn_horizon_controller_v24 import PNNHorizonControllerV24
from team_code.pnn_pid_bridge import PNNHiPADPIDBridge
from team_code.visualize import draw_bboxes3d
from leaderboard.autoagents import autonomous_agent

PROJECT_ROOT = pathlib.Path(
    os.environ.get("HIPAD_PNN_ROOT", pathlib.Path(__file__).resolve().parents[3])
).resolve()
MAP_ROOT = os.environ.get("MAP_ROOT", str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT))
from pnn_temporal_alignment import (
    ALIGNMENT_VERSION,
    HIPAD_MOTION_DT,
    align_hipad_motion_future,
)
from hipad_pnn_adapter import (
    PNNAdapterConfig,
    PNNOptimizerAdapter,
    select_left_right_lane_boundaries,
    select_left_right_lane_boundaries_v2,
)


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

LIDAR2IMG = {
    'CAM_FRONT': np.array([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                           [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                           [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_LEFT': np.array([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_FRONT_RIGHT': np.array([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                 [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                 [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK': np.array([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                          [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                          [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                          [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_LEFT': np.array([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                               [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                               [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                               [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

    'CAM_BACK_RIGHT': np.array([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
}

LIDAR2CAM = {
    'CAM_FRONT': np.array([[1., 0., 0., 0.],
                           [0., 0., -1., -0.24],
                           [0., 1., 0., -1.19],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[0.57357644, 0.81915204, 0., -0.22517331],
                                [0., 0., -1., -0.24],
                                [-0.81915204, 0.57357644, 0., -0.82909407],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[0.57357644, -0.81915204, 0., 0.22517331],
                                 [0., 0., -1., -0.24],
                                 [0.81915204, 0.57357644, 0., -0.82909407],
                                 [0., 0., 0., 1.]]),

    'CAM_BACK': np.array([[-1., 0., 0., 0.],
                          [0., 0., -1., -0.24],
                          [0., -1., 0., -1.61],
                          [0., 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[-0.34202014, 0.93969262, 0., -0.25388956],
                               [0., 0., -1., -0.24],
                               [-0.93969262, -0.34202014, 0., -0.49288953],
                               [0., 0., 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[-0.34202014, -0.93969262, 0., 0.25388956],
                                [0., 0., -1., -0.24],
                                [0.93969262, -0.34202014, 0., -0.49288953],
                                [0., 0., 0., 1.]])
}

CAM2IMG = {
    'CAM_FRONT': np.array([[1142.51841 , 0., 800., 0.],
                           [0., 1142.51841, 450., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]]),

    'CAM_FRONT_LEFT': np.array([[1142.51840553, 0., 800., 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]]),

    'CAM_FRONT_RIGHT': np.array([[1142.51841061, 0., 800, 0.],
                                 [0., 1142.51841, 450, 0.],
                                 [0., 0. ,1., 0.],
                                 [0., 0. ,0., 1.]]),

    'CAM_BACK': np.array([[560.166031, 0., 800., 0.],
                          [0. ,560.166031, 450., 0.],
                          [0. , 0., 1., 0.],
                          [0. , 0., 0., 1.]]),

    'CAM_BACK_LEFT': np.array([[1142.51840683, 0., 800., 0.],
                               [0., 1142.51841, 450., 0.],
                               [0., 0. , 1., 0.],
                               [0., 0. , 0., 1.]]),

    'CAM_BACK_RIGHT': np.array([[1142.51841041, 0., 800, 0.],
                                [0., 1142.51841, 450., 0.],
                                [0., 0. , 1., 0.],
                                [0., 0. , 0., 1.]])
}

LIDAR2EGO = np.array([[0., 1., 0., -0.39],
                      [-1., 0., 0., 0.],
                      [0., 0., 1., 1.84],
                      [0., 0., 0., 1.]])

unreal2cam = np.array([[0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0],
                               [0.0, 548.993771650447, 256.0, 0],
                               [0.0, 0.0, 1.0, 0],
                               [0, 0, 0, 1.0]])

topdown_extrinsics = np.array([[0.0, -0.0, -1.0, 50.0],
                               [0.0, 1.0, -0.0, 0.0],
                               [1.0, -0.0, 0.0, -0.0],
                               [0.0, 0.0, 0.0, 1.0]])

CAMERA = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

def get_entry_point():
    return 'SparseAgent'


class SparseAgent(autonomous_agent.AutonomousAgent):
    def sensors(self):
        sensors = [
            # camera rgb
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT'
            },
            # imu
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU'
            },
            # gps
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS'
            },
            # speed
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED'
            },
        ]
        if IS_BENCH2DRIVE:
            sensors += [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.0, 'y': 0.0, 'z': 60.0,
                    'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                    'width': 512, 'height': 512, 'fov': 5 * 10.0,
                    'id': 'bev'
                }]
        return sensors

    def setup(self, path_to_conf_file):
        self.device = "cuda"
        self.track = autonomous_agent.Track.SENSORS

        self.step = -1
        self.steer_step = 0
        self.last_steer = 0
        self.last_moving_step = -1
        self.last_moving_status = 0
        self.frame_rate = 20

        self.ckpt_path = path_to_conf_file.split('+')[1]
        self.config_path = path_to_conf_file.split('+')[0]
        self.save_name = path_to_conf_file.split('+')[-1]

        self.pidcontroller = PIDController(
            turn_KP=1.,
            turn_KI=0.75,
            turn_KD=0.0,
            turn_n=10,
            speed_KP=5.0,
            speed_KI=0.5,
            speed_KD=1.0,
            speed_n=10,
            waypoint_time=float(os.environ.get("PNN_PID_WAYPOINT_TIME", "0.5")))
        self.pnn_control_mode = os.environ.get("PNN_CONTROL_MODE", "pid").strip().lower()
        if self.pnn_control_mode not in {
            "pid", "horizon", "horizon_v24", "bridge", "hybrid"
        }:
            raise ValueError(
                "PNN_CONTROL_MODE must be 'pid', 'bridge', 'hybrid', "
                "'horizon', or 'horizon_v24'"
            )
        self.pnn_pid_bridge = PNNHiPADPIDBridge(dt=0.1, temporal_hz=2.0, spatial_step_m=2.0)
        self.pnn_horizon_controller = PNNHorizonController(
            trajectory_pid=self.pidcontroller,
            dt=0.1,
            accel_decay_steps=float(os.environ.get("PNN_CONTROL_DECAY_STEPS", "6.0")),
            steer_feedforward_blend=float(
                os.environ.get("PNN_STEER_FEEDFORWARD_BLEND", "0.35")
            ),
            enable_dynamic_guard=os.environ.get("PNN_DYNAMIC_BRAKE_GUARD", "1") == "1",
            use_trajectory_pid=os.environ.get("PNN_HORIZON_USE_LEGACY_PID", "0") == "1",
            rolling_throttle=float(os.environ.get("PNN_ROLLING_THROTTLE", "0.10")),
            restart_throttle=float(os.environ.get("PNN_RESTART_THROTTLE", "0.22")),
            restart_speed_threshold=float(
                os.environ.get("PNN_RESTART_SPEED_THRESHOLD", "0.35")
            ),
            restart_target_speed=float(
                os.environ.get("PNN_RESTART_TARGET_SPEED", "0.45")
            ),
            restart_accel_threshold=float(
                os.environ.get("PNN_RESTART_ACCEL_THRESHOLD", "0.10")
            ),
            stop_target_speed=float(
                os.environ.get("PNN_STOP_TARGET_SPEED", "0.45")
            ),
            stop_accel_ceiling=float(
                os.environ.get("PNN_STOP_ACCEL_CEILING", "0.0")
            ),
            hard_guard_steps=int(
                os.environ.get("PNN_HARD_GUARD_STEPS", "3")
            ),
            long_risk_throttle_cap=float(
                os.environ.get("PNN_LONG_RISK_THROTTLE_CAP", "0.25")
            ),
            guard_rear_tolerance=float(
                os.environ.get("PNN_GUARD_REAR_TOLERANCE", "0.25")
            ),
            guard_release_decay=float(
                os.environ.get("PNN_GUARD_RELEASE_DECAY", "0.55")
            ),
            enable_ttc_guard=os.environ.get("PNN_TTC_GUARD", "0") == "1",
            ttc_hard_seconds=float(os.environ.get("PNN_TTC_HARD_SECONDS", "1.25")),
            ttc_soft_seconds=float(os.environ.get("PNN_TTC_SOFT_SECONDS", "2.25")),
            ttc_safety_buffer=float(os.environ.get("PNN_TTC_SAFETY_BUFFER", "2.5")),
            ttc_low_speed_release=float(
                os.environ.get("PNN_TTC_LOW_SPEED_RELEASE", "0.30")
            ),
            ttc_creep_clearance=float(
                os.environ.get("PNN_TTC_CREEP_CLEARANCE", "5.0")
            ),
        )
        self.pnn_horizon_controller_v24 = PNNHorizonControllerV24(
            dt=0.1,
            steer_feedforward_weight=float(
                os.environ.get("PNN_V24_STEER_FF_WEIGHT", "0.70")
            ),
            speed_feedback_gain=float(
                os.environ.get("PNN_V24_SPEED_FEEDBACK_GAIN", "0.80")
            ),
            steer_rate_limit=float(
                os.environ.get("PNN_V24_STEER_RATE_LIMIT", "0.12")
            ),
            throttle_rise_limit=float(
                os.environ.get("PNN_V24_THROTTLE_RISE_LIMIT", "0.08")
            ),
            brake_release_limit=float(
                os.environ.get("PNN_V24_BRAKE_RELEASE_LIMIT", "0.20")
            ),
            enable_ttc_guard=os.environ.get("PNN_V24_TTC_GUARD", "1") == "1",
        )
        self.pnn_vehicle_physics_calibrated = False

        self.wall_start = time.time()
        self.initialized = False
        self.use_bgr_img = True
        self.data_aug_conf = None

        cfg = Config.fromfile(self.config_path)
        cfg.model.head.onedecoder_head.with_close_loop = True # align training frequency
        if hasattr(cfg, 'data_aug_conf'):
            self.data_aug_conf = cfg.data_aug_conf
        if hasattr(cfg.model.head, 'evaluate_bench2dive'):
            cfg.model.head.evaluate_bench2dive = False
        if hasattr(cfg, "plugin") and hasattr(cfg, "plugin_dir") and cfg.plugin:
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split("/")
            _module_path = _module_dir[0]

            for m in _module_dir[1:]:
                _module_path = _module_path + "." + m
            print(_module_path)
            plg_lib = importlib.import_module(_module_path)

        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(self.model)
        if self.ckpt_path is not None:
            checkpoint = load_checkpoint(self.model, self.ckpt_path, map_location='cpu')

        self.model.cuda()
        self.model.eval()
        self.inference_only_pipeline = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
            if inference_only_pipeline["type"] not in ['LoadMultiViewImageFromFilesInCeph', 'LoadMultiViewImageFromFiles']:
                self.inference_only_pipeline.append(inference_only_pipeline)
        self.inference_only_pipeline = Compose(self.inference_only_pipeline)

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0
        self.lat_ref, self.lon_ref = 42.0, 2.0

        control = carla.VehicleControl()
        control.steer = 0.0
        control.brake = 0.0
        control.throttle = 0.0

        self.prev_control = control
        self.prev_control_cache = []
        pnn_ckpt_default = str(PROJECT_ROOT / "checkpoints" / "pnn_control.pth")
        self.pnn_adapter = PNNOptimizerAdapter(
            PNNAdapterConfig(
                device=os.environ.get("PNN_ADAPTER_DEVICE", "cuda:0"),
                stats_path=os.environ.get(
                    "PNN_STATS_PATH",
                    str(PROJECT_ROOT / "checkpoints" / "pnn_stats.pt"),
                ),
                control_ckpt_path=os.environ.get(
                    "PNN_CONTROL_CKPT",
                    pnn_ckpt_default,
                ),
                weight_ckpt_path=os.environ.get(
                    "PNN_WEIGHT_CKPT",
                    pnn_ckpt_default,
                ),
                use_weight_net=os.environ.get("PNN_USE_WEIGHT_NET", "0") != "0",
                use_theseus_refine=False,
                weight_temperature=float(os.environ.get("PNN_WEIGHT_TEMPERATURE", "0.7")),
                weight_delta_max=(1.3, 1.4, 1.1, 1.4, 1.2, 1.2, 1.5, 1.4),
                output_forward_offset=float(os.environ.get("PNN_OUTPUT_FORWARD_OFFSET", "0.0")),
                reference_forward_offset=float(os.environ.get("PNN_REFERENCE_FORWARD_OFFSET", "0.0")),
                coord_convention=os.environ.get("PNN_COORD_CONVENTION", "pnn_xy"),
                stats_quantile_low=float(os.environ.get("PNN_STATS_QUANTILE_LOW", "0.005")),
                stats_quantile_high=float(os.environ.get("PNN_STATS_QUANTILE_HIGH", "0.995")),
                clamp_normalized_inputs=os.environ.get("PNN_CLAMP_NORMALIZED_INPUTS", "1") == "1",
                control_min_accel=float(os.environ.get("PNN_CONTROL_DECODE_MIN_ACCEL", "-10.0")),
                control_max_accel=float(os.environ.get("PNN_CONTROL_DECODE_MAX_ACCEL", "10.0")),
                control_max_steer=float(os.environ.get("PNN_CONTROL_DECODE_MAX_STEER", "1.066")),
            )
        )

        self.is_visualize = os.environ.get("PNN_VISUALIZE", "0") == "1"
        self.visualize_interval = int(os.environ.get("PNN_VISUALIZE_INTERVAL", "2"))
        self.visualize_ego_trajs = os.environ.get("PNN_VISUALIZE_EGO_TRAJS", "all").lower()
        self.visualize_draw_lanes = os.environ.get("PNN_VISUALIZE_DRAW_LANES", "1") == "1"

        string = pathlib.Path(os.environ['ROUTES']).stem + '_'
        string += self.save_name
        self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
        self.save_path.mkdir(parents=True, exist_ok=True)
        if self.is_visualize:
            (self.save_path / 'metas').mkdir(exist_ok=True)
            (self.save_path / 'images').mkdir(exist_ok=True)

        self.lidar2cam = LIDAR2CAM
        self.lidar2img = LIDAR2IMG
        self.lidar2ego = LIDAR2EGO
        self.cam2img = CAM2IMG
        self.coor2topdown = unreal2cam @ topdown_extrinsics
        self.coor2topdown = topdown_intrinsics @ self.coor2topdown

    def init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = ((lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)) -
                       math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA
                       * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180)
                       * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True
        self.metric_info = {}

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in CAMERA:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        pos = self.gps_to_location(gps)
        waypoint_routes = list(self._route_planner.run_step(pos))
        navigation_xy = [np.asarray(route_item[0], dtype=np.float32) for route_item in waypoint_routes[1:]]
        if not navigation_xy and waypoint_routes:
            navigation_xy = [np.asarray(waypoint_routes[0][0], dtype=np.float32)]

        if len(waypoint_routes) >= 3:
            target_xy = waypoint_routes[1][0]
            target_xy_next = waypoint_routes[2][0]
            command = waypoint_routes[0][1]
            command_next = waypoint_routes[1][1]
        elif len(waypoint_routes) == 2:
            target_xy = target_xy_next = waypoint_routes[1][0]
            command = command_next = waypoint_routes[0][1]
        else:
            target_xy = target_xy_next = waypoint_routes[0][0]
            command = command_next = waypoint_routes[0][1]

        if (math.isnan(compass) == True):  # It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'bev': bev,
            'speed': speed,
            'compass': compass,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'target_xy': target_xy,
            'target_xy_next': target_xy_next,
            'navigation_xy': navigation_xy,
            'command': command,
            'command_next': command_next
        }

        return result

    def destroy(self):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = (int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH)
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat + 90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self.init()
        tick_data = self.tick(input_data)
        inputs = {}
        inputs['img'] = []
        inputs['folder'] = ''
        inputs['scene_token'] = ''
        inputs['lidar2img'] = []
        inputs['lidar2cam'] = []
        inputs['frame_idx'] = 0
        inputs['timestamp'] = self.step / self.frame_rate
        for cam in CAMERA:
            inputs['lidar2img'].append(self.lidar2img[cam])
            inputs['lidar2cam'].append(self.lidar2cam[cam])
            img = tick_data['imgs'][cam]
            if self.use_bgr_img:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            inputs['img'].append(img)
        inputs['lidar2img'] = np.stack(inputs['lidar2img'], axis=0)
        inputs['lidar2cam'] = np.stack(inputs['lidar2cam'], axis=0)
        inputs['aug_config'] = self.get_augmentation()

        # data
        ego_pos = [tick_data['pos'][0], -tick_data['pos'][1]]
        target_xy = [tick_data['target_xy'][0], -tick_data['target_xy'][1]]
        target_xy_next = [tick_data['target_xy_next'][0], -tick_data['target_xy_next'][1]]
        navigation_xy = np.asarray(
            [[point[0], -point[1]] for point in tick_data['navigation_xy']],
            dtype=np.float32,
        )

        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi / 2
        ego_theta_degree = ego_theta / np.pi * 180
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        ego_speed = tick_data['speed']
        acceleration = tick_data['acceleration']
        ego_accel = [acceleration[0], -acceleration[1], acceleration[2]]
        angular_velocity = -tick_data['angular_velocity']

        # status
        custom_status = np.zeros(6)
        custom_status[0] = ego_speed
        custom_status[1:3] = ego_accel[:2]
        custom_status[3:5] = angular_velocity[:2]
        custom_status[5] = self.pid_metadata['steer'] if hasattr(self, 'pid_metadata') else 0
        inputs['custom_status'] = np.array(custom_status, np.float32)

        # command
        command = tick_data['command']
        command_next = tick_data['command_next']
        if command < 0: command = 4
        command -= 1
        command_onehot = np.zeros(6, np.float32)
        command_onehot[command] = 1
        inputs['command'] = command
        inputs['gt_ego_fut_cmd'] = command_onehot
    
        # target point
        target_xy = np.array([target_xy[0] - ego_pos[0], target_xy[1] - ego_pos[1]])
        target_xy_next = np.array([target_xy_next[0] - ego_pos[0], target_xy_next[1] - ego_pos[1]])

        theta_to_lidar = raw_theta
        rotation_matrix = np.array([[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                                    [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]])

        target_point = np.array(rotation_matrix @ target_xy, dtype=np.float32)
        target_point_next = np.array(rotation_matrix @ target_xy_next, dtype=np.float32)
        navigation_points = np.asarray(
            [rotation_matrix @ (point - np.asarray(ego_pos[:2], dtype=np.float32)) for point in navigation_xy],
            dtype=np.float32,
        )

        inputs['target_point'] = target_point
        inputs['target_point_next'] = target_point_next
        inputs['navigation_points'] = navigation_points

        # metas
        ego2world = np.eye(4)
        ego2world[0:3, 0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2, 3] = ego_pos
        
        lidar2global = ego2world @ self.lidar2ego
        inputs['l2g_r_mat'] = lidar2global[0:3, 0:3]
        inputs['l2g_t'] = lidar2global[0:3, 3]
        inputs['lidar2global'] = lidar2global
        image_h, image_w = self.data_aug_conf['final_dim']
        inputs['image_wh'] = np.array([[image_w, image_h] for _ in CAMERA], dtype=np.float32)

        inputs = self.inference_only_pipeline(inputs)
        inputs = mm_collate_to_batch_form([inputs], samples_per_gpu=1)
        for key, value in inputs.items():
            if isinstance(value, DataContainer):
                inputs[key] = value.data[0]
            elif isinstance(value[0], DataContainer):
                inputs[key] = value[0].data
            else:
                inputs[key] = value
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        outputs = self.model(
            img=inputs['img'],
            img_metas=inputs['img_metas'],
            projection_mat=inputs['projection_mat'],
            gt_ego_fut_cmd=inputs['gt_ego_fut_cmd'],
            image_wh=inputs['image_wh'],
            timestamp=inputs['timestamp'],
            target_point=inputs['target_point'],
            rescale=True,
            return_loss=False,
        )

        # control
        # PNN was trained/evaluated with 2 Hz HiP-AD plans: six points at 0.5 s.
        # Keep the temporal scale aligned before sending route targets to PNN.
        plan_temp_name = os.environ.get("PNN_HIPAD_PLAN_KEY", "plan_temp_2hz")
        fallback_plan_names = ["plan_speed_2hz", "plan_temp_2hz", "plan_speed_5hz", "plan_temp_5hz"]
        plan_spat_name = 'plan_spat_2m'

        pred_temp_traj = None
        selected_plan_name = None
        for candidate_name in [plan_temp_name] + [x for x in fallback_plan_names if x != plan_temp_name]:
            if candidate_name in outputs[0]['img_bbox']:
                pred_temp_traj = outputs[0]['img_bbox'][candidate_name].cpu().numpy()
                outputs[0]['img_bbox']['temporal_planning'] = outputs[0]['img_bbox'][candidate_name]
                selected_plan_name = candidate_name
                break
        if pred_temp_traj is None:
            raise RuntimeError(f"PNN wrapper could not find any HiP-AD plan in {fallback_plan_names}.")

        pred_spat_traj = None
        if plan_spat_name in outputs[0]['img_bbox']:
            pred_spat_traj = outputs[0]['img_bbox'][plan_spat_name].cpu().numpy()
            outputs[0]['img_bbox']['spatial_planning'] = outputs[0]['img_bbox'][plan_spat_name]

        ped_agents, veh_agents = self.extract_pnn_agents(outputs)
        lane_points, lane_selector_info = self.extract_pnn_lane_points(
            outputs,
            reference_plan=pred_temp_traj,
        )
        pnn_route_source = os.environ.get("PNN_ROUTE_SOURCE", "hipad_plan").strip().lower()
        if pnn_route_source == "navigation":
            pnn_result = self.pnn_adapter.refine_navigation_route(
                navigation_points=navigation_points,
                ego_speed=float(ego_speed),
                hipad_plan=pred_temp_traj,
                spatial_plan=pred_spat_traj,
                ped_agents=ped_agents,
                veh_agents=veh_agents,
                lane_points=lane_points,
                navigation_min_speed=float(os.environ.get("PNN_NAV_MIN_SPEED", "1.0")),
                navigation_max_speed=float(os.environ.get("PNN_NAV_MAX_SPEED", "15.0")),
                navigation_distance_scale=os.environ.get("PNN_NAV_DISTANCE_SCALE", "1.0"),
                navigation_interpolation=os.environ.get("PNN_NAV_INTERPOLATION", "spline"),
            )
        elif pnn_route_source == "hipad_plan":
            pnn_result = self.pnn_adapter.refine_hipad_plan(
                hipad_plan=pred_temp_traj,
                ego_speed=float(ego_speed),
                spatial_plan=pred_spat_traj,
                ped_agents=ped_agents,
                veh_agents=veh_agents,
                lane_points=lane_points,
            )
        else:
            raise ValueError(f"Unsupported PNN_ROUTE_SOURCE={pnn_route_source!r}")
        pnn_traj = pnn_result['final_planning']

        if self.pnn_control_mode in {"horizon", "horizon_v24"} and not self.pnn_vehicle_physics_calibrated:
            try:
                if not hasattr(self, "hero_actor") or self.hero_actor is None:
                    self.get_hero()
                physics = self.hero_actor.get_physics_control()
                steer_angles = [
                    float(wheel.max_steer_angle)
                    for wheel in physics.wheels
                    if float(wheel.max_steer_angle) > 1e-3
                ]
                if steer_angles:
                    max_steer_angle = float(np.deg2rad(max(steer_angles)))
                    self.pnn_horizon_controller.max_steer_angle = max_steer_angle
                    self.pnn_horizon_controller_v24.max_steer_angle = max_steer_angle
                self.pnn_vehicle_physics_calibrated = True
            except Exception as exc:
                print(f"[PNN] vehicle physics calibration deferred: {exc}")

        bridge_metadata = {}
        if self.pnn_control_mode == "horizon":
            steer_traj, throttle_traj, brake_traj, metadata_traj = (
                self.pnn_horizon_controller.control(
                    dense_trajectory=pnn_result['dense_trajectory'],
                    control_horizon=pnn_result['control'],
                    speed=ego_speed,
                    target_point=target_point,
                    veh_agents=veh_agents,
                    ped_agents=ped_agents,
                )
            )
        elif self.pnn_control_mode == "horizon_v24":
            steer_traj, throttle_traj, brake_traj, metadata_traj = (
                self.pnn_horizon_controller_v24.control(
                    dense_trajectory=pnn_result['dense_trajectory'],
                    control_horizon=pnn_result['control'],
                    speed=ego_speed,
                    target_point=target_point,
                    veh_agents=veh_agents,
                    ped_agents=ped_agents,
                )
            )
        elif self.pnn_control_mode in {"bridge", "hybrid"}:
            bridge_temp, bridge_spat, bridge_metadata = self.pnn_pid_bridge.build(
                pnn_result['dense_trajectory']
            )
            bridge_metadata["bridge_temporal_plan"] = bridge_temp.tolist()
            bridge_metadata["bridge_spatial_plan"] = bridge_spat.tolist()
            steer_traj, throttle_traj, brake_traj, metadata_traj = (
                self.pidcontroller.control_pid(
                    bridge_temp, bridge_spat, ego_speed, target_point
                )
            )
            if self.pnn_control_mode == "hybrid":
                throttle_traj, brake_traj, guard_metadata = (
                    self.pnn_horizon_controller.guard_pid_control(
                        dense_trajectory=pnn_result['dense_trajectory'],
                        throttle=throttle_traj,
                        brake=brake_traj,
                        veh_agents=veh_agents,
                        ped_agents=ped_agents,
                        speed=ego_speed,
                    )
                )
                bridge_metadata.update(guard_metadata)
        else:
            steer_traj, throttle_traj, brake_traj, metadata_traj = (
                self.pidcontroller.control_pid(
                    pnn_traj, pnn_traj, ego_speed, target_point
                )
            )
        if brake_traj < 0.05: brake_traj = 0.0
        if throttle_traj > brake_traj: brake_traj = 0.0

        control = carla.VehicleControl()
        control.steer = np.clip(float(steer_traj), -1, 1)
        control.throttle = np.clip(float(throttle_traj), 0, 0.75)
        control.brake = np.clip(float(brake_traj), 0, 1)

        self.pid_metadata = metadata_traj
        self.pid_metadata.update(bridge_metadata)
        self.pid_metadata['agent'] = 'only_traj'
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_traj'] = float(brake_traj)
        self.pid_metadata['plan_temp'] = pred_temp_traj.tolist()
        self.pid_metadata['plan_temp_key'] = selected_plan_name
        self.pid_metadata['plan_spat'] = pred_spat_traj.tolist() if pred_spat_traj is not None else None
        self.pid_metadata['pnn_plan'] = pnn_traj.tolist()
        self.pid_metadata['pnn_control'] = pnn_result['control'].tolist()
        self.pid_metadata['pnn_dense_trajectory'] = pnn_result['dense_trajectory'].tolist()
        self.pid_metadata['pnn_raw_dense_trajectory'] = pnn_result.get('raw_dense_trajectory', pnn_result['dense_trajectory']).tolist()
        self.pid_metadata['pnn_initial_plan'] = pnn_result['initial_final_planning'].tolist()
        self.pid_metadata['pnn_raw_initial_plan'] = pnn_result.get('raw_initial_final_planning', pnn_result['initial_final_planning']).tolist()
        self.pid_metadata['pnn_output_forward_offset'] = float(pnn_result.get('output_forward_offset', 0.0))
        self.pid_metadata['pnn_reference_forward_offset'] = float(pnn_result.get('reference_forward_offset', 0.0))
        self.pid_metadata['pnn_effective_output_forward_offset'] = float(pnn_result.get('effective_output_forward_offset', 0.0))
        self.pid_metadata['pnn_coord_convention'] = pnn_result.get('coord_convention', os.environ.get("PNN_COORD_CONVENTION", "hipad_xy"))
        self.pid_metadata['pnn_cost_weights'] = pnn_result['cost_weights'].tolist()
        self.pid_metadata['pnn_route_source'] = pnn_result['route_source']
        self.pid_metadata['pnn_route_targets'] = pnn_result['route_targets'].tolist()
        self.pid_metadata['pnn_actor_alignment_version'] = ALIGNMENT_VERSION
        self.pid_metadata['pnn_control_mode'] = self.pnn_control_mode
        self.pid_metadata['pnn_carla_max_steer_angle_rad'] = float(
            self.pnn_horizon_controller.max_steer_angle
        )
        self.pid_metadata['pnn_actor_motion_source_dt'] = HIPAD_MOTION_DT
        self.pid_metadata['pnn_navigation_points'] = navigation_points.tolist()
        self.pid_metadata['pnn_num_ped_agents'] = len(ped_agents)
        self.pid_metadata['pnn_num_veh_agents'] = len(veh_agents)
        self.pid_metadata['pnn_ped_agents'] = self.serialize_pnn_agents(ped_agents)
        self.pid_metadata['pnn_veh_agents'] = self.serialize_pnn_agents(veh_agents)
        self.pid_metadata['pnn_lane_points'] = lane_points.tolist()
        self.pid_metadata['pnn_lane_points_shape'] = list(lane_points.shape)
        self.pid_metadata['pnn_lane_selector'] = lane_selector_info
        self.pid_metadata['command'] = command
        self.pid_metadata['target_point'] = [float(target_point[0]), float(target_point[1])]

        metric_info = self.get_metric_info()
        metric_info["pnn_control_debug"] = {
            key: self.pid_metadata[key]
            for key in (
                "controller",
                "pnn_accel_first",
                "pnn_accel_near",
                "pnn_accel_all",
                "pnn_forecast_decel",
                "pnn_accel_command",
                "pnn_target_speed_05",
                "pnn_target_speed_10",
                "pnn_dynamic_brake_floor",
                "pnn_dynamic_long_risk",
                "pnn_dynamic_min_distance",
                "pnn_dynamic_risk_time",
                "pnn_dynamic_risk_kind",
                "pnn_dynamic_overlap_time",
                "pnn_dynamic_ttc",
                "pnn_dynamic_required_decel",
                "pnn_dynamic_low_speed_release",
                "pnn_dynamic_ttc_guard",
                "pnn_num_veh_agents",
                "pnn_num_ped_agents",
                "pnn_dense_steer",
                "throttle",
                "brake",
                "steer",
            )
            if key in self.pid_metadata
        }
        self.metric_info[self.step] = metric_info

        outfile = open(self.save_path / 'metric_info.json', 'w')
        json.dump(self.metric_info, outfile, indent=4)
        outfile.close()

        if self.is_visualize and self.step % self.visualize_interval == 0:
            self.visualize(
                tick_data,
                inputs,
                outputs,
                pred_temp_traj,
                target_point,
                pnn_traj,
                lane_points,
            )

        self.prev_control = control
        if len(self.prev_control_cache) == 10:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        return control

    def visualize(
        self,
        tick_data,
        input_batch,
        output_batch,
        pred_planning,
        target_point,
        pnn_planning=None,
        lane_points=None,
    ):
        rw, rh = 960//2, 540//2

        pred_planning = np.concatenate([np.array([[0, 0]]), pred_planning])
        if pnn_planning is not None:
            pnn_planning = np.concatenate([np.array([[0, 0]]), pnn_planning])
        pnn_only_visualize = self.visualize_ego_trajs in {"pnn", "pnn_only", "pnn-only"}
        if pnn_only_visualize and pnn_planning is not None:
            pred_planning = pnn_planning
            pnn_planning = None

        with_spatial_planning = False
        if 'spatial_planning' in output_batch[0]['img_bbox']:
            with_spatial_planning = True
            spatial_planning = output_batch[0]['img_bbox']['spatial_planning'].cpu().numpy()
            spatial_planning = np.concatenate([np.array([[0, 0]]), spatial_planning])
        if pnn_only_visualize:
            with_spatial_planning = False

        ### plot img
        imgs = []
        for cam in CAMERA:
            img = tick_data['imgs'][cam]

            # draw agent box on image
            if 'boxes_3d' in output_batch[0]['img_bbox']:
                pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)

                img = draw_bboxes3d(img, pred_bboxes3d,
                                    intrinsic=self.cam2img[cam],
                                    extrinsic=self.lidar2cam[cam],
                                    color=(0, 0, 255))

            # draw ego traj on image
            if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                # spatial traj
                if with_spatial_planning:
                    coord3d = np.concatenate([spatial_planning,
                                              np.ones_like(pred_planning[:, :1]) * -1.8,
                                              np.ones_like(pred_planning[:, :1])], axis=1)
                    coord2d = (self.lidar2img[cam] @ coord3d.T).T
                    coord2d = coord2d[coord2d[:, 2] > 1e-5]
                    coord2d = coord2d[:, :2] / coord2d[:, 2:3]
                    img = self.draw_trajectory(img, coord2d, line_color=(0, 200, 255), line_thickness=10,
                                               point_color=(0, 200, 255), point_thickness=8, point_radius=8)

                # temporal traj
                coord3d = np.concatenate([pred_planning,
                                          np.ones_like(pred_planning[:, :1]) * -1.8,
                                          np.ones_like(pred_planning[:, :1])], axis=1)
                coord2d = (self.lidar2img[cam] @ coord3d.T).T
                coord2d = coord2d[coord2d[:, 2] > 1e-5]
                coord2d = coord2d[:, :2] / coord2d[:, 2:3]
                img = self.draw_trajectory(img, coord2d, line_color=(255, 0, 0), line_thickness=10,
                                           point_color=(255, 0, 0), point_thickness=8, point_radius=8)

                # PNN-refined temporal traj
                if pnn_planning is not None:
                    coord3d = np.concatenate([pnn_planning,
                                              np.ones_like(pnn_planning[:, :1]) * -1.8,
                                              np.ones_like(pnn_planning[:, :1])], axis=1)
                    coord2d = (self.lidar2img[cam] @ coord3d.T).T
                    coord2d = coord2d[coord2d[:, 2] > 1e-5]
                    coord2d = coord2d[:, :2] / coord2d[:, 2:3]
                    img = self.draw_trajectory(img, coord2d, line_color=(255, 0, 255), line_thickness=10,
                                               point_color=(255, 0, 255), point_thickness=8, point_radius=8)


            # draw target point on image
            if cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                cmd_coord3d = np.concatenate([target_point,
                                              np.ones_like(target_point[:1]) * -1.8,
                                              np.ones_like(target_point[:1])], axis=0)
                cmd_coord2d = (self.lidar2img[cam] @ cmd_coord3d.T).T
                cmd_coord2d = cmd_coord2d[:2] / np.abs(cmd_coord2d[2:3])
                img = cv2.circle(img, (int(cmd_coord2d[0]), int(cmd_coord2d[1])), 7, (255, 105, 120), 6)

            # text and resize
            img = cv2.putText(img, cam, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            img = cv2.resize(img, (rw, rh))
            imgs.append(img)

        ### plot bev
        bev_img = tick_data['bev']
        bev_dict = self.sensors()[-1]
        if bev_dict['id']=='bev':
            ## agent
            # draw agent box on bev
            if 'boxes_3d' in output_batch[0]['img_bbox']:
                pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)
                for idx, (pred_bbox3d, pred_label3d) in enumerate(zip(pred_bboxes3d, pred_labels3d)):
                    box = self.convert_bev_bbox(pred_bbox3d, bev_dict)
                    cv2.polylines(bev_img, [box], isClosed=True, color=(0, 0, 255), thickness=1)

                    # draw agent motion traj on bev
                    if pred_trajs3d is not None and pred_label3d in [0, 1, 2, 3]:
                        pred_traj = np.concatenate([pred_bbox3d[:2][None], pred_trajs3d[idx]])
                        bev_coord = self.convert_bev_coord(pred_traj, bev_dict)
                        bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 0), line_thickness=1,
                                                       point_color=(0, 200, 0), point_thickness=1, point_radius=2)

            ### ego
            # draw ego box on bev
            box = self.convert_bev_bbox((0, 0, 0, 1.84, 4.89, 1.49, 0), bev_dict)
            cv2.polylines(bev_img, [box], isClosed=True, color=(255, 255, 0), thickness=1)

            # draw ego spatial traj
            if with_spatial_planning:
                bev_coord = self.convert_bev_coord(spatial_planning, bev_dict)
                bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(0, 200, 255), line_thickness=1,
                                               point_color=(0, 200, 255), point_thickness=1, point_radius=2)
            # draw ego temporal traj
            bev_coord = self.convert_bev_coord(pred_planning, bev_dict)
            bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(255, 0, 0), line_thickness=1,
                                           point_color=(255, 0, 0), point_thickness=1, point_radius=2)
            # draw PNN-refined ego temporal traj
            if pnn_planning is not None:
                bev_coord = self.convert_bev_coord(pnn_planning, bev_dict)
                bev_img = self.draw_trajectory(bev_img, bev_coord, line_color=(255, 0, 255), line_thickness=1,
                                               point_color=(255, 0, 255), point_thickness=1, point_radius=2)

            # PNN map input: first two selected vectors are left/right
            # boundaries; remaining vectors are lower-priority candidates.
            if self.visualize_draw_lanes and lane_points is not None:
                lane_colors = [(255, 255, 0), (255, 165, 0)]
                for lane_index, lane in enumerate(np.asarray(lane_points)):
                    bev_coord = self.convert_bev_coord(lane, bev_dict)
                    color = lane_colors[lane_index] if lane_index < 2 else (140, 140, 140)
                    thickness = 2 if lane_index < 2 else 1
                    cv2.polylines(
                        bev_img,
                        [np.asarray(bev_coord, dtype=np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=thickness,
                    )

            # draw bev target point
            bev_coord = self.convert_bev_coord(target_point, bev_dict)
            cv2.circle(bev_img, (int(bev_coord[0]), int(bev_coord[1])), 3, (255, 105, 120), 2)

        # text and resize
        cmd_str = str(tick_data['command']).split('.')[-1]
        bev_img = cv2.putText(bev_img, cmd_str, (15, bev_dict['height']-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if pnn_only_visualize:
            bev_img = cv2.putText(bev_img, "PNN", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        else:
            bev_img = cv2.putText(bev_img, "HiPAD", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        if not pnn_only_visualize:
            bev_img = cv2.putText(bev_img, "PNN", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # text and resize
        # bev_img = cv2.putText(bev_img, "cmd:{}".format(self.pid_metadata['command']), (15, bev_dict['height']-95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "speed:{:.2f}".format(self.pid_metadata['speed']), (15, bev_dict['height']-75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "steer:{:.2f}".format(self.pid_metadata['steer']), (15, bev_dict['height']-55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "throttle:{:.2f}".format(self.pid_metadata['throttle']), (15, bev_dict['height']-35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, "brake:{:.2f}".format(self.pid_metadata['brake']), (15, bev_dict['height']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        # bev_img = cv2.putText(bev_img, str(tick_data['command']), (15, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        bev_img = cv2.resize(bev_img, (rh*2, rh*2))

        # merge
        front = imgs[0]
        front_left = imgs[1]
        front_right = imgs[2]
        back = imgs[3]
        back_left = imgs[4]
        back_right = imgs[5]

        line1 = np.hstack([front_left, front, front_right])
        line2 = np.hstack([back_right, back, back_left])
        merge_img = np.vstack([line1, line2])
        merge_img = np.hstack([merge_img, bev_img])

        frame = self.step

        Image.fromarray(merge_img).save(self.save_path / 'images' / ('%04d.jpg' % frame))

        outfile = open(self.save_path / 'metas' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    @staticmethod
    def serialize_pnn_agents(agents):
        serialized = []
        for agent in agents:
            serialized.append(
                {
                    "x": float(agent["x"]),
                    "y": float(agent["y"]),
                    "yaw": float(agent["yaw"]),
                    "speed": float(agent["speed"]),
                    "future": np.asarray(agent["future"], dtype=np.float32).tolist(),
                    "goal": np.asarray(agent["goal"], dtype=np.float32).tolist(),
                    "goal_time": float(agent["goal_time"]),
                    "alignment_version": ALIGNMENT_VERSION,
                }
            )
        return serialized

    def get_bboxes(self, output_batch):
        pred_bboxes3d = copy.deepcopy(output_batch[0]['img_bbox']['boxes_3d'].cpu().numpy())
        pred_scores3d = copy.deepcopy(output_batch[0]['img_bbox']['scores_3d'].cpu().numpy())
        pred_labels3d = copy.deepcopy(output_batch[0]['img_bbox']['labels_3d'].cpu().numpy())

        pred_mask3d = pred_scores3d > 0.3
        pred_bboxes3d = pred_bboxes3d[pred_mask3d]
        pred_labels3d = pred_labels3d[pred_mask3d]

        pred_trajs3d = None
        if 'trajs_3d' in output_batch[0]['img_bbox']:
            pred_trajs3d = copy.deepcopy(output_batch[0]['img_bbox']['trajs_3d'].cpu().numpy())
            pred_trajs_score = copy.deepcopy(output_batch[0]['img_bbox']['trajs_score'].cpu().numpy())

            pred_trajs3d = pred_trajs3d[pred_mask3d]
            pred_trajs_score = pred_trajs_score[pred_mask3d]

            # select top-one
            pred_trajs3d = pred_trajs3d[np.arange(len(pred_trajs3d)), pred_trajs_score.argmax(-1)]

        return pred_bboxes3d, pred_labels3d, pred_trajs3d

    def extract_pnn_agents(self, output_batch):
        pred_bboxes3d, pred_labels3d, pred_trajs3d = self.get_bboxes(output_batch)
        # An empty scene is valid, especially during the first few frames after
        # loading a route. The PNN adapter pads empty pedestrian/vehicle sets
        # and masks them out, so do not crash merely because HiP-AD has no
        # detections above the confidence threshold.
        if len(pred_bboxes3d) == 0:
            return [], []
        if pred_trajs3d is None:
            raise RuntimeError("PNN wrapper requires HiP-AD agent motion output 'trajs_3d'.")

        ped_agents = []
        veh_agents = []
        for bbox, label, future in zip(pred_bboxes3d, pred_labels3d, pred_trajs3d):
            label = int(label)
            future = np.asarray(future, dtype=np.float32)
            if future.ndim != 2 or future.shape[-1] != 2 or future.shape[0] == 0:
                continue

            x = float(bbox[0])
            y = float(bbox[1])
            yaw = float(bbox[6]) if len(bbox) > 6 else math.pi / 2

            velocity_xy = bbox[7:9] if len(bbox) >= 9 else None
            aligned_future = align_hipad_motion_future(
                current_xy=(x, y),
                future=future,
                source_dt=HIPAD_MOTION_DT,
                measured_velocity_xy=velocity_xy,
                max_speed=5.0 if label == 7 else 20.0,
            )
            if velocity_xy is not None:
                speed = float(np.linalg.norm(velocity_xy))
            elif aligned_future.shape[0] >= 2:
                speed = float(np.linalg.norm(aligned_future[1] - aligned_future[0]) / 0.5)
            else:
                speed = 0.0
            target_xy = aligned_future[-1]
            target_t = 3.0

            agent = {
                "x": x,
                "y": y,
                "yaw": yaw,
                "speed": speed,
                "future": aligned_future,
                "goal": target_xy,
                "goal_time": target_t,
            }

            # HiP-AD B2D labels:
            # 0 car, 1 van, 2 truck, 3 bicycle, 7 pedestrian.
            if label == 7:
                ped_agents.append(agent)
            elif label in (0, 1, 2, 3):
                veh_agents.append(agent)

        return ped_agents, veh_agents

    def extract_pnn_lane_points(self, output_batch, reference_plan=None):
        img_bbox = output_batch[0]['img_bbox']
        if 'vectors' not in img_bbox:
            raise RuntimeError("PNN wrapper requires HiP-AD map output 'vectors'.")

        vectors = img_bbox['vectors']
        scores = img_bbox.get('scores', None)
        labels = img_bbox.get('labels', None)

        vector_list = [np.asarray(vec, dtype=np.float32) for vec in vectors]
        if len(vector_list) < 2:
            raise RuntimeError("PNN wrapper requires at least two HiP-AD map vectors for lane_points.")

        if scores is None:
            score_arr = np.ones((len(vector_list),), dtype=np.float32)
        else:
            score_arr = np.asarray(scores, dtype=np.float32)

        if labels is None:
            label_arr = np.zeros((len(vector_list),), dtype=np.int64)
        else:
            label_arr = np.asarray(labels, dtype=np.int64)

        selector = os.environ.get("PNN_LANE_SELECTOR", "v1").strip().lower()
        if selector in {"v2", "reference_plan_v2", "ref_v2"}:
            lane_points, selector_info = select_left_right_lane_boundaries_v2(
                vectors=vector_list,
                scores=score_arr,
                labels=label_arr,
                reference_plan=reference_plan,
                num_lanes=10,
                num_points=20,
                return_info=True,
            )
        elif selector in {"v1", "legacy", "default"}:
            lane_points = select_left_right_lane_boundaries(
                vectors=vector_list,
                scores=score_arr,
                labels=label_arr,
                reference_plan=reference_plan,
                num_lanes=10,
                num_points=20,
            )
            selector_info = {"selector": "legacy_v1", "fallback": False}
        else:
            raise ValueError(f"Unsupported PNN_LANE_SELECTOR={selector!r}")
        if lane_points.shape[0] < 2:
            raise RuntimeError("PNN wrapper could not build two valid lane boundary/map vectors.")
        return lane_points, selector_info

    def draw_trajectory(self, image, coord2d, line_color=(255, 0, 0), line_thickness=2,
                        point_color=(255, 0, 0), point_thickness=2, point_radius=None):

        for i in range(len(coord2d) - 1):
            point_start = coord2d[i]
            point_end = coord2d[i + 1]

            point_start = (int(point_start[0]), int(point_start[1]))
            point_end = (int(point_end[0]), int(point_end[1]))

            cv2.line(image, point_start, point_end, line_color, line_thickness)

            # draw points
            if point_radius is not None:
                cv2.circle(image, point_start, point_radius, point_color, point_thickness)
                cv2.circle(image, point_end, point_radius, point_color, point_thickness)

        return image

    def convert_bev_bbox(self, bbox, bev_dict):
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        center = (bev_width / 2 + bbox[0] * bev_ratio_w,
                  bev_height / 2 - bbox[1] * bev_ratio_h)
        size = (bbox[3] * bev_ratio_w, bbox[4] * bev_ratio_h)
        angle = - bbox[6] / np.pi * 180

        rect = (center, size, angle)
        bev_bbox = cv2.boxPoints(rect)
        bev_bbox = np.int0(bev_bbox)

        return bev_bbox

    def convert_bev_coord(self, coord3d, bev_dict):
        bev_cam_z = bev_dict['z']
        bev_cam_fov = bev_dict['fov']
        bev_width = bev_dict['width']
        bev_height = bev_dict['height']

        bev_range = np.tan(bev_cam_fov / 2 / 180 * np.pi) * bev_cam_z * 2

        bev_ratio_w = bev_width / bev_range
        bev_ratio_h = bev_height / bev_range

        if len(coord3d.shape) == 1:
            coord_x = coord3d[0:1] * bev_ratio_w
            coord_y = coord3d[1:2] * bev_ratio_h
        else:
            coord_x = coord3d[:, 0:1] * bev_ratio_w
            coord_y = coord3d[:, 1:2] * bev_ratio_h

        offset_u = bev_width / 2 + coord_x
        offset_v = bev_height / 2 - coord_y
        coord2d = np.concatenate([offset_u, offset_v], axis=-1)

        return coord2d
