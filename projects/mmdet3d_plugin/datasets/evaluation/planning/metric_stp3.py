'''
calculate planner metric same as stp3
'''
import numpy as np
import torch
import cv2
import copy
import matplotlib.pyplot as plt
from skimage.draw import polygon
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R

ego_width, ego_length = 1.85, 4.084

MAX_ABS_MAG_JERK = 8.37      # [m/s^3], Bench2Drive smoothness threshold
MAX_ABS_LAT_ACCEL = 4.89     # [m/s^2]
MAX_LON_ACCEL = 2.40         # [m/s^2]
MIN_LON_ACCEL = -4.05        # [m/s^2]
MAX_ABS_YAW_ACCEL = 1.93     # [rad/s^2]
MAX_ABS_LON_JERK = 4.13      # [m/s^3]
MAX_ABS_YAW_RATE = 0.95      # [rad/s]

class PlanningMetric():
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        # Keep this aligned with det_class_names in hipad_b2d_stage2.py.
        # gt_agent_feats[..., 27] stores the compact 0-based class index.
        det_class_names = (
            "car",
            "van",
            "truck",
            "bicycle",
            "traffic_sign",
            "traffic_cone",
            "traffic_light",
            "pedestrian",
            "others",
        )
        self.category_index = {
            "vehicle": [det_class_names.index(name) for name in ("car", "van", "truck", "bicycle")],
            "human": [det_class_names.index("pedestrian")],
        }

        # self.n_future = n_future

        # self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # self.add_state("L2", default=torch.zeros(self.n_future),dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @staticmethod
    def _as_numpy(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _cross_2d(a, b):
        return a[0] * b[1] - a[1] * b[0]

    @staticmethod
    def _point_segment_distance(p, a, b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    @classmethod
    def _segments_intersect(cls, p1, p2, q1, q2, eps=1e-8):
        r = p2 - p1
        s = q2 - q1
        denom = cls._cross_2d(r, s)
        qp = q1 - p1
        if abs(denom) < eps:
            return (
                cls._point_segment_distance(p1, q1, q2) < eps
                or cls._point_segment_distance(p2, q1, q2) < eps
                or cls._point_segment_distance(q1, p1, p2) < eps
                or cls._point_segment_distance(q2, p1, p2) < eps
            )
        t = cls._cross_2d(qp, s) / denom
        u = cls._cross_2d(qp, r) / denom
        return -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps

    @classmethod
    def _segment_distance(cls, p1, p2, q1, q2):
        if cls._segments_intersect(p1, p2, q1, q2):
            return 0.0
        return min(
            cls._point_segment_distance(p1, q1, q2),
            cls._point_segment_distance(p2, q1, q2),
            cls._point_segment_distance(q1, p1, p2),
            cls._point_segment_distance(q2, p1, p2),
        )

    @staticmethod
    def _points_in_convex_polygon(points, polygon, eps=1e-6):
        if len(points) == 0:
            return np.zeros((0,), dtype=bool)
        signs = []
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            edge = b - a
            rel = points - a[None]
            signs.append(edge[0] * rel[:, 1] - edge[1] * rel[:, 0])
        signs = np.stack(signs, axis=1)
        return np.logical_or(np.all(signs >= -eps, axis=1), np.all(signs <= eps, axis=1))

    def _ego_corners_at(self, xy, yaw):
        # Match evaluate_single_coll: ego footprint is slightly shifted forward.
        local = np.array([
            [-self.H / 2.0 + 0.5, self.W / 2.0],
            [self.H / 2.0 + 0.5, self.W / 2.0],
            [self.H / 2.0 + 0.5, -self.W / 2.0],
            [-self.H / 2.0 + 0.5, -self.W / 2.0],
        ], dtype=np.float32)
        rot = np.array(
            [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
            dtype=np.float32,
        )
        return local @ rot.T + xy[None]

    def _select_valid_map_polyline(self, pts):
        pts = self._as_numpy(pts).astype(np.float32)
        if pts.ndim == 3:
            # gt_map_pts is usually [num_perm, num_pts, 2]. For open polylines
            # the first two permutations are forward/backward and later entries
            # are padded with 1e5. Pick the permutation with most valid points.
            valid = np.isfinite(pts).all(axis=-1) & (np.abs(pts).max(axis=-1) < 1e4)
            perm_idx = int(np.argmax(valid.sum(axis=1)))
            pts = pts[perm_idx]
        valid = np.isfinite(pts).all(axis=-1) & (np.abs(pts).max(axis=-1) < 1e4)
        pts = pts[valid]
        if len(pts) <= 1:
            return None
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-3:
                keep.append(i)
        pts = pts[keep]
        if len(pts) <= 1:
            return None
        return pts

    def evaluate_lane_edge_coll(
            self,
            traj,
            gt_map_pts,
            gt_map_labels,
            lane_boundary_labels=(1, 2),
            collision_margin=0.05,
        ):
        """Evaluate ego footprint collision with GT lane boundaries.

        Bench2Drive map labels in the current config are:
        0=Broken, 1=Solid, 2=SolidSolid, 3=Center. Broken is a lane divider
        that may be legally crossed, so the default CCR proxy only uses solid
        lane markings and excludes Center.
        """
        traj_np = self._as_numpy(traj).astype(np.float32)
        map_pts_np = self._as_numpy(gt_map_pts)
        labels_np = self._as_numpy(gt_map_labels)
        device = traj.device if torch.is_tensor(traj) else torch.device("cpu")
        n_future = traj_np.shape[0]
        if map_pts_np is None or labels_np is None or len(map_pts_np) == 0:
            return torch.zeros(n_future, dtype=torch.bool, device=device)

        labels_np = labels_np.reshape(-1)
        polylines = []
        for pts, label in zip(map_pts_np, labels_np):
            if int(label) not in lane_boundary_labels:
                continue
            polyline = self._select_valid_map_polyline(pts)
            if polyline is not None:
                polylines.append(polyline[:, :2])
        if not polylines:
            return torch.zeros(n_future, dtype=torch.bool, device=device)

        collisions = np.zeros(n_future, dtype=bool)
        # HiP-AD local coordinates are x=right, y=forward. A stationary ego
        # therefore starts at pi/2, not zero. Keep that heading through tiny
        # localization jitter instead of rotating the footprint arbitrarily.
        prev_yaw = np.pi / 2
        for t in range(n_future):
            if n_future == 1:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            elif t == 0:
                direction = traj_np[1, :2] - traj_np[0, :2]
            else:
                direction = traj_np[t, :2] - traj_np[t - 1, :2]
            if np.linalg.norm(direction) > 0.1:
                prev_yaw = float(np.arctan2(direction[1], direction[0]))
            corners = self._ego_corners_at(traj_np[t, :2], prev_yaw)
            rect_segments = [
                (corners[i], corners[(i + 1) % len(corners)])
                for i in range(len(corners))
            ]

            hit = False
            for polyline in polylines:
                if np.any(self._points_in_convex_polygon(polyline, corners)):
                    hit = True
                    break
                for i in range(len(polyline) - 1):
                    a, b = polyline[i], polyline[i + 1]
                    for c, d in rect_segments:
                        if self._segment_distance(a, b, c, d) <= collision_margin:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            collisions[t] = hit

        return torch.from_numpy(collisions).to(device=device)

    @staticmethod
    def _angle_normalize_np(x):
        return np.arctan2(np.sin(x), np.cos(x))

    def compute_openloop_comfort(self, traj, dt=0.5):
        """Compute an open-loop comfort proxy from planned 2D positions.

        This is not the closed-loop CARLA comfortness metric because open-loop
        predictions do not contain actual IMU/control traces. It mirrors the
        same physical quantities as closely as possible from a 2Hz xy plan.
        """
        traj_np = self._as_numpy(traj).astype(np.float32)
        if traj_np.ndim == 3:
            traj_np = traj_np[0]
        traj_np = traj_np[:, :2]
        if len(traj_np) < 2:
            return {
                "comfort_score": 1.0,
                "max_abs_lon_accel": 0.0,
                "max_abs_lat_accel": 0.0,
                "max_abs_jerk": 0.0,
                "max_abs_lon_jerk": 0.0,
                "max_abs_yaw_rate": 0.0,
                "max_abs_yaw_accel": 0.0,
            }

        delta = np.diff(traj_np, axis=0)
        speed = np.linalg.norm(delta, axis=1) / float(dt)
        heading = np.arctan2(delta[:, 1], delta[:, 0])

        if len(speed) >= 2:
            lon_acc = np.diff(speed) / float(dt)
            max_abs_lon_accel = float(np.max(np.abs(lon_acc)))
        else:
            lon_acc = np.zeros((0,), dtype=np.float32)
            max_abs_lon_accel = 0.0

        if len(lon_acc) >= 2:
            lon_jerk = np.diff(lon_acc) / float(dt)
            max_abs_lon_jerk = float(np.max(np.abs(lon_jerk)))
            max_abs_jerk = max_abs_lon_jerk
        else:
            lon_jerk = np.zeros((0,), dtype=np.float32)
            max_abs_lon_jerk = 0.0
            max_abs_jerk = 0.0

        if len(heading) >= 2:
            yaw_rate = self._angle_normalize_np(np.diff(heading)) / float(dt)
            speed_mid = speed[1:]
            lat_acc = speed_mid * yaw_rate
            max_abs_yaw_rate = float(np.max(np.abs(yaw_rate)))
            max_abs_lat_accel = float(np.max(np.abs(lat_acc)))
        else:
            yaw_rate = np.zeros((0,), dtype=np.float32)
            max_abs_yaw_rate = 0.0
            max_abs_lat_accel = 0.0

        if len(yaw_rate) >= 2:
            yaw_accel = np.diff(yaw_rate) / float(dt)
            max_abs_yaw_accel = float(np.max(np.abs(yaw_accel)))
        else:
            max_abs_yaw_accel = 0.0

        lon_acc_ok = True
        if len(lon_acc) > 0:
            lon_acc_ok = bool(np.all((lon_acc >= MIN_LON_ACCEL) & (lon_acc <= MAX_LON_ACCEL)))
        lon_jerk_ok = max_abs_lon_jerk <= MAX_ABS_LON_JERK
        mag_jerk_ok = max_abs_jerk <= MAX_ABS_MAG_JERK
        lat_acc_ok = max_abs_lat_accel <= MAX_ABS_LAT_ACCEL
        yaw_rate_ok = max_abs_yaw_rate <= MAX_ABS_YAW_RATE
        yaw_accel_ok = max_abs_yaw_accel <= MAX_ABS_YAW_ACCEL
        comfort_score = float(
            lon_acc_ok
            and lat_acc_ok
            and mag_jerk_ok
            and lon_jerk_ok
            and yaw_rate_ok
            and yaw_accel_ok
        )

        return {
            "comfort_score": comfort_score,
            "max_abs_lon_accel": max_abs_lon_accel,
            "max_abs_lat_accel": max_abs_lat_accel,
            "max_abs_jerk": max_abs_jerk,
            "max_abs_lon_jerk": max_abs_lon_jerk,
            "max_abs_yaw_rate": max_abs_yaw_rate,
            "max_abs_yaw_accel": max_abs_yaw_accel,
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian
    
    def get_birds_eye_view_label(
            self,
            gt_agent_boxes,
            gt_agent_feats
        ):
        '''
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9) 
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        '''
        T = 6
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.cpu().numpy()  #(N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., :T*2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T*2:T*3].reshape(-1, 6)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
        gt_agent_fut_yaw = gt_agent_feats[..., T*3+10:T*4+10].reshape(-1, 6, 1)
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:,6:7] = -1 * (gt_agent_boxes[:, 6:7] + np.pi/2) # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]
        
        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    # Filter out all non vehicle instances
                    category_index = int(gt_agent_feats[0,i][27])
                    agent_length, agent_width = gt_agent_boxes[i][4], gt_agent_boxes[i][3]
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a,y_a,yaw_a,agent_length, agent_width]
                    if (category_index in self.category_index['vehicle']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if (category_index in self.category_index['human']):
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)
        
        # vis for debug
        # plt.figure('debug')
        # for i in range(T):
        #     plt.subplot(2,T,i+1)
        #     plt.imshow(segmentation[i])
        #     plt.subplot(2,T,i+1+T)
        #     plt.imshow(pedestrian[i])
        # plt.savefig('outputs/debug/car_ped_occ.jpg')
        # plt.close()

        return segmentation, pedestrian
    
    def _get_poly_region_in_image(self,param):
        lidar2cv_rot = np.array([[1,0], [0,-1]])
        x_a,y_a,yaw_a,agent_length, agent_width = param
        trans_a = np.array([[x_a,y_a]]).T
        rot_mat_a = np.array([[np.cos(yaw_a), -np.sin(yaw_a)],
                                [np.sin(yaw_a), np.cos(yaw_a)]])
        agent_corner = np.array([
            [agent_length/2, -agent_length/2, -agent_length/2, agent_length/2],
            [agent_width/2, agent_width/2, -agent_width/2, -agent_width/2]]) #(2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a #(2,4)
        # convert to cv frame
        agent_corner_cv2 = (np.matmul(lidar2cv_rot, agent_corner_lidar) \
            - self.bev_start_position[:2,None] + self.bev_resolution[:2,None] / 2.0).T / self.bev_resolution[:2] #(4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2


    def evaluate_single_coll(self, traj, segmentation, input_gt):
        '''
        traj: torch.Tensor (n_future, 2)
            自车lidar系为轨迹参考系
                ^ y
                |
                | 
                0------->
                        x
        segmentation: torch.Tensor (n_future, 200, 200)
        '''
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.],
            [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # 轨迹坐标系转换为:
        #  ^ x
        #  |
        #  | 
        #  0-------> y
        trajs_ = copy.deepcopy(trajs)
        trajs_[:,:,[0,1]] = trajs_[:,:,[1,0]] # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:,:,0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:,:,1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())
        
        # vis for debug
        # obs_occ = copy.deepcopy(segmentation)
        # ego_occ = torch.zeros_like(obs_occ)
        # for t in range(n_future):
        #     rr = r[t]
        #     cc = c[t]
        #     I = np.logical_and(
        #         np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
        #         np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
        #     )
        #     ego_occ[t, rr[I], cc[I]]=1
        
        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2,6,i+1)
        #     plt.imshow(obs_occ[i])
        #     plt.subplot(2,6,i+7)
        #     plt.imshow(ego_occ[i])
        # if input_gt:
        #     plt.savefig('outputs/debug/occ_metric_stp3_gt.jpg')
        # else:
        #     plt.savefig('outputs/debug/occ_metric_stp3_pred.jpg')
        # plt.close()

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(
            self, 
            trajs, 
            gt_trajs, 
            segmentation
        ):
        '''
        trajs: torch.Tensor (B, n_future, 2)
            自车lidar系为轨迹参考系
            ^ y
            |
            | 
            0------->
                    x
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        '''
        gt_trajs = gt_trajs.to(device=trajs.device)
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            xx, yy = trajs[i,:,0], trajs[i, :, 1]
            # lidar系下的轨迹转换到图片坐标系下
            # Occupancy labels are built on CPU while model trajectories may
            # be on CUDA during distributed evaluation. Keep every tensor
            # used for occupancy indexing on the occupancy device.
            xi = ((-self.bx[0]/2 - yy) / self.dx[0]).long().to(segmentation.device)
            yi = ((-self.bx[1]/2 + xx) / self.dx[1]).long().to(segmentation.device)

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            )
            #import pdb;pdb.set_trace()

            ti = torch.arange(n_future, device=segmentation.device)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()

            box_coll = self.evaluate_single_coll(
                trajs[i], segmentation[i], input_gt=False
            ).to(segmentation.device)
            obj_box_coll_sum += box_coll.long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        '''
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        '''
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )
        
        return ade

    # def update(self, trajs, gt_trajs, segmentation):
    #     '''
    #     trajs: torch.Tensor (B, n_future, 3)
    #     gt_trajs: torch.Tensor (B, n_future, 3)
    #     segmentation: torch.Tensor (B, n_future, 200, 200)
    #     '''
    #     assert trajs.shape == gt_trajs.shape
    #     L2 = self.compute_L2(trajs, gt_trajs)
    #     obj_coll_sum, obj_box_coll_sum = self.evaluate_coll(trajs[:,:,:2], gt_trajs[:,:,:2], segmentation)

    #     if torch.isnan(L2).max().item():
    #         debug = 1
    #     else:
    #         self.obj_col += obj_coll_sum
    #         self.obj_box_col += obj_box_coll_sum
    #         self.L2 += L2.sum(dim=0)
    #         if torch.isnan(self.L2).max().item():
    #             debug=1
    #         self.total +=len(trajs)


    # def compute(self):
    #     return {
    #         'obj_col': self.obj_col / self.total,
    #         'obj_box_col': self.obj_box_col / self.total,
    #         'L2' : self.L2 / self.total
    #     }
