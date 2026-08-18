import torch
import torch.nn as nn
import torch.nn.functional as F
# -------- 矩形距离模块 --------
class RectangleDistance(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def get_corners(self, cx, cy, theta, length, width):
        """
        cx,cy,theta,length,width: [B]
        return: [B,4,2]
        """
        # Clockwise perimeter order is required by edges+roll below. The old
        # order crossed the rectangle diagonally, producing invalid SAT axes
        # and many false-positive overlaps.
        local = torch.tensor(
            [[-0.5, -0.5],
             [-0.5,  0.5],
             [ 0.5,  0.5],
             [ 0.5, -0.5]],
            device=cx.device, dtype=cx.dtype
        ).unsqueeze(0)                           # [1,4,2]

        # 确保 length/width 是 Tensor，形状 [B]
        if not torch.is_tensor(length):
            length = torch.full_like(cx, float(length))
        if not torch.is_tensor(width):
            width  = torch.full_like(cx, float(width))
        length = length.to(device=cx.device, dtype=cx.dtype)
        width  = width.to(device=cx.device, dtype=cx.dtype)

        scale = torch.stack([length, width], dim=-1).unsqueeze(1)  # [B,1,2]
        pts = local * scale                                        # [B,4,2]

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        R = torch.stack([
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t,  cos_t], dim=-1)
        ], dim=-2)                                                # [B,2,2]

        pts = torch.matmul(pts, R.transpose(-1, -2))               # [B,4,2]
        pts = pts + torch.stack([cx, cy], dim=-1).unsqueeze(1)     # [B,4,2]
        return pts                                                 # ←← 必须 return

    def project_interval(self, pts, axis):
        """
        pts: [B,4,2], axis: [B,2]
        return: (min, max) each [B]
        """
        axis_norm = axis / (axis.norm(dim=-1, keepdim=True) + self.eps)  # [B,2]
        proj = (pts * axis_norm.unsqueeze(1)).sum(-1)                    # [B,4]
        return proj.min(-1).values, proj.max(-1).values

    def forward(self, state_a, shape_a, state_b, shape_b):
        cx1, cy1, th1 = state_a[:, 0], state_a[:, 1], state_a[:, 2]
        cx2, cy2, th2 = state_b[:, 0], state_b[:, 1], state_b[:, 2]
        l1 = torch.as_tensor(shape_a[0], device=cx1.device, dtype=cx1.dtype).expand_as(cx1)
        w1 = torch.as_tensor(shape_a[1], device=cx1.device, dtype=cx1.dtype).expand_as(cx1)
        l2 = torch.as_tensor(shape_b[0], device=cx1.device, dtype=cx1.dtype).expand_as(cx2)
        w2 = torch.as_tensor(shape_b[1], device=cx1.device, dtype=cx1.dtype).expand_as(cx2)

        rect1 = self.get_corners(cx1, cy1, th1, l1, w1)  # [B,4,2]
        rect2 = self.get_corners(cx2, cy2, th2, l2, w2)

        edges1 = rect1 - torch.roll(rect1, shifts=1, dims=1)  # [B,4,2]
        edges2 = rect2 - torch.roll(rect2, shifts=1, dims=1)  # [B,4,2]
        axes = torch.cat([edges1, edges2], dim=1)             # [B,8,2]
        axes = torch.stack([-axes[...,1], axes[...,0]], -1)   # [B,8,2]

        # === 向量化投影 ===
        # rect1, rect2 : [B,4,2] -> [B,8,4,2] broadcast
        axes_exp = axes.unsqueeze(2)          # [B,8,1,2]
        rect1_exp = rect1.unsqueeze(1)        # [B,1,4,2]
        rect2_exp = rect2.unsqueeze(1)
        axes_norm = axes / (axes.norm(dim=-1, keepdim=True)+self.eps) # [B,8,2]

        proj1 = (rect1_exp*axes_norm.unsqueeze(2)).sum(-1)    # [B,8,4]
        proj2 = (rect2_exp*axes_norm.unsqueeze(2)).sum(-1)
        min1,max1 = proj1.min(-1).values, proj1.max(-1).values  # [B,8]
        min2,max2 = proj2.min(-1).values, proj2.max(-1).values

        gap1 = min2 - max1
        gap2 = min1 - max2
        gaps = torch.maximum(gap1, gap2)                      # [B,8]

        sep_or_overlap,_ = gaps.max(dim=1)
        return F.relu(sep_or_overlap), F.relu(-sep_or_overlap)




# -------- Soft λ for secondary constraints (learned by Adam) --------
class SoftConstraintLambdas(nn.Module):
    """可训练的软约束 λ 参数，交给 Adam 更新"""
    def __init__(self, init_dict=None):
        super().__init__()
        if init_dict is None:
            init_dict = {
                'ctrl_ego': 0.5,
                'ctrl_p':   0.5,
                'ctrl_v':   0.5,
                'vel_p':    1.0,
                'vel_v':    1.0,
                'l1':       1e-2,
            }
        self.lambdas = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.exp(torch.tensor([val], dtype=torch.float32)) - 1.0))
            for name, val in init_dict.items()
        })

    def get(self, name):
        return F.softplus(self.lambdas[name], beta=1.0)

    def forward(self):
        return {name: F.softplus(param, beta=1.0) for name, param in self.lambdas.items()}

# -------- Hard λ for lane boundary (manual dual update) --------
class LaneBoundaryLagrangianLoss(nn.Module):
    def __init__(self, init_lambda, softplus_beta=10.0, margin=0.3,
                 max_penalty=100.0, trainable=True, lambda_cap=100000.0):
        super().__init__()
        self.margin = margin
        self.beta = softplus_beta
        self.max_penalty = max_penalty
        self.lambda_cap = float(lambda_cap)

        # softplus^-1(init_lambda)
        init_log = torch.log(torch.exp(torch.tensor([init_lambda], dtype=torch.float32)) - 1.0)
        self.log_lambda = nn.Parameter(init_log.clone(), requires_grad=trainable)
    
    @property
    def lambda_val(self):
        return F.softplus(self.log_lambda, beta=1.0).clamp(max=self.lambda_cap)

    def compute_constraint_violation(self, traj_points, lane_boundary,
                                     key_ts=(0, 5, 10, 15, 20, 24), delta_t=5):
        """
        返回每个样本的 soft violation（不乘 λ），基于左右最近车道边界的方向一致性。
        traj_points: [B, T, 2]
        lane_boundary: [B, L, N, 2]
        """
        B, T, _ = traj_points.shape
        B2, L, N, _ = lane_boundary.shape
        assert B2 == B, "batch size mismatch"
        violations_per_batch = []

        for t_idx in key_ts:
            if t_idx < 0 or t_idx >= T: continue
            # The last block starts at 2.5 s (index 24) and reaches index 29,
            # so the complete 3 s rollout receives an explicit constraint.
            if t_idx + delta_t >= T: continue

            ref_point = traj_points[:, t_idx, :]                   # [B,2]
            ref_point_exp = ref_point.unsqueeze(1).unsqueeze(2).expand(-1, L, N, 2)

            dists = torch.norm(ref_point_exp - lane_boundary, dim=-1)   # [B,L,N]
            min_dist_vals, min_idx_pts = dists.min(dim=-1)              # [B,L], [B,L]

            idx_expand = min_idx_pts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            nearest_pts = torch.gather(lane_boundary, 2, idx_expand).squeeze(2)  # [B,L,2]

            idx_next = torch.clamp(min_idx_pts + 1, max=N-1)
            idx_prev = torch.clamp(min_idx_pts - 1, min=0)
            idx_next_expand = idx_next.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            idx_prev_expand = idx_prev.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
            next_pts = torch.gather(lane_boundary, 2, idx_next_expand).squeeze(2)
            prev_pts = torch.gather(lane_boundary, 2, idx_prev_expand).squeeze(2)

            mask_use_next = (min_idx_pts < (N - 1)).unsqueeze(-1)
            tangent = torch.where(mask_use_next.expand(-1, -1, 2),
                                  next_pts - nearest_pts,
                                  nearest_pts - prev_pts)              # [B,L,2]

            vec = ref_point.unsqueeze(1).expand(-1, L, 2) - nearest_pts # [B,L,2]
            cross_z = tangent[..., 0] * vec[..., 1] - tangent[..., 1] * vec[..., 0]
            cross_sign = torch.sign(cross_z)                             # [B,L]

            mask_left = cross_sign > 0
            mask_right = cross_sign < 0

            inf_tensor = torch.full_like(min_dist_vals, float('inf'))
            dist_left_masked = torch.where(mask_left, min_dist_vals, inf_tensor)
            dist_right_masked = torch.where(mask_right, min_dist_vals, inf_tensor)

            left_min_vals, idx_left_sel = dist_left_masked.min(dim=-1)     # [B]
            right_min_vals, idx_right_sel = dist_right_masked.min(dim=-1)  # [B]

            idx_left_sel_expand = idx_left_sel.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
            idx_right_sel_expand = idx_right_sel.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 2)
            nearest_left_pts = torch.gather(nearest_pts, 1, idx_left_sel_expand).squeeze(1)   # [B,2]
            nearest_right_pts = torch.gather(nearest_pts, 1, idx_right_sel_expand).squeeze(1) # [B,2]

            ref_vec_left  = ref_point - nearest_left_pts
            ref_vec_right = ref_point - nearest_right_pts

            for dt_step in range(1, delta_t + 1):
                cur_point = traj_points[:, t_idx + dt_step, :]  # [B,2]
                vec_left  = cur_point - nearest_left_pts
                vec_right = cur_point - nearest_right_pts

                cos_left  = F.cosine_similarity(vec_left,  ref_vec_left,  dim=-1, eps=1e-8)
                cos_right = F.cosine_similarity(vec_right, ref_vec_right, dim=-1, eps=1e-8)

                mask_valid_left  = left_min_vals  < float('inf')
                mask_valid_right = right_min_vals < float('inf')

                margin = 0.0
                sp0 = F.softplus(torch.tensor(0.0, device=cos_left.device) * self.beta)

                delta_left  = margin - cos_left
                delta_right = margin - cos_right

                vio_left  = F.softplus(self.beta * delta_left)  - sp0
                vio_right = F.softplus(self.beta * delta_right) - sp0

                vio_left  = torch.clamp_min(vio_left,  0.0) * mask_valid_left.float()
                vio_right = torch.clamp_min(vio_right, 0.0) * mask_valid_right.float()

                violations_per_batch.append(vio_left + vio_right)  # [B]

        if len(violations_per_batch) == 0:
            return torch.zeros(B, device=traj_points.device)

        all_vios = torch.stack(violations_per_batch, dim=0)  # [K,B]
        return all_vios.mean(dim=0)                           # [B]

    def forward(self, traj_points, lane_boundary):
        g = self.compute_constraint_violation(traj_points, lane_boundary)  # [B]
        penalty = self.lambda_val * g
        return torch.clamp(penalty, max=self.max_penalty).mean()

# -------- Hard λ for safety distance --------
class SafetyConstraintLoss(nn.Module):
    def __init__(self, init_lambda, min_dist=1.5, lambda_cap=1e6,
                 ego_shape=(4.5, 2.0), ped_shape=(0.6, 0.6), veh_shape=(4.5, 2.0),
                 check_frames=(4,9,14,19,24,29), topk=3,
                 time_weights=(2.0, 1.8, 1.5, 1.2, 1.0, 0.8),
                 veh_veh_weight=0.05):
        super().__init__()
        self.min_dist = float(min_dist)
        self.lambda_cap = float(lambda_cap)
        self.log_lambda = nn.Parameter(
            torch.log(torch.exp(torch.tensor([init_lambda], dtype=torch.float32)) - 1.0),
            requires_grad=True
        )
        self.rect_dist = RectangleDistance()
        self.ego_shape = ego_shape
        self.ped_shape = ped_shape
        self.veh_shape = veh_shape
        self.check_frames = tuple(check_frames)
        self.topk = int(topk)
        self.time_weights = tuple(float(x) for x in time_weights)
        self.veh_veh_weight = float(veh_veh_weight)

    @property
    def lambda_val(self):
        return F.softplus(self.log_lambda, beta=1.0).clamp(max=self.lambda_cap)

    def _pair_ego_vs_group_dense(self, ego_t, group_t, mask, shape_group):
        """
        ego_t:   [B,4]
        group_t: [B,N,4]
        mask:    [B,N] bool
        return: per-agent safety-buffer violation [B,N], zero for invalid agents.

        This is closer to raw ACR than the previous mean-over-all-pairs loss:
        a single dangerous actor should not be diluted by many safe actors.
        """
        B, N, _ = group_t.shape
        out = torch.zeros((B, N), device=group_t.device, dtype=group_t.dtype)
        if N == 0:
            return out
        valid = mask.bool()
        if not valid.any():
            return out
        ego_expand = ego_t.unsqueeze(1).expand(-1, N, -1)
        ego_sel = ego_expand[valid]
        grp_sel = group_t[valid]
        separation, penetration = self.rect_dist(
            ego_sel, self.ego_shape, grp_sel, shape_group
        )
        signed_separation = separation - penetration
        out[valid] = F.relu(self.min_dist - signed_separation).to(group_t.dtype)
        return out

    def _pair_veh_vs_veh(self, veh_t, veh_mask):
        """
        Non-ego collision is retained only as a tiny regularizer. Raw ACR is an
        ego-object metric, so this term must not dominate ControlNet gradients.
        """
        B, Nv, _ = veh_t.shape
        if Nv < 2:
            return torch.zeros(B, device=veh_t.device, dtype=veh_t.dtype)
        vi = veh_t.unsqueeze(2).expand(-1, Nv, Nv, -1)
        vj = veh_t.unsqueeze(1).expand(-1, Nv, Nv, -1)
        valid = veh_mask.unsqueeze(2).bool() & veh_mask.unsqueeze(1).bool()
        triu = torch.triu(torch.ones(Nv, Nv, device=veh_t.device, dtype=torch.bool), diagonal=1)
        valid = valid & triu.unsqueeze(0)
        if not valid.any():
            return torch.zeros(B, device=veh_t.device, dtype=veh_t.dtype)
        a = vi[valid]
        b = vj[valid]
        _, pen = self.rect_dist(a, self.veh_shape, b, self.veh_shape)
        batch_idx = torch.arange(B, device=veh_t.device).view(B, 1, 1).expand(B, Nv, Nv)[valid]
        pen_max = torch.zeros(B, device=veh_t.device, dtype=veh_t.dtype)
        # scatter_reduce_ is not available on some torch 1.x builds; emulate max.
        for bi in range(B):
            vals = pen[batch_idx == bi]
            if vals.numel() > 0:
                pen_max[bi] = vals.max().to(veh_t.dtype)
        return pen_max

    def _aggregate_topk(self, values):
        """values [B,K]; return top-k mean with max-like behaviour."""
        if values.numel() == 0 or values.shape[1] == 0:
            return torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        k = min(max(self.topk, 1), values.shape[1])
        topk = torch.topk(values, k=k, dim=1).values
        # 70% max + 30% top-k mean: stable gradient but still ACR-like.
        return 0.7 * topk[:, 0] + 0.3 * topk.mean(dim=1)

    def compute_constraint_violation(self, traj_ego, traj_peds, traj_vehs, ped_mask, veh_mask):
        """
        traj_*: ego [B,T,4], peds [B,Np,T,4], vehs [B,Nv,T,4]
        mask:   ped [B,Np], veh [B,Nv]

        Returns [B] ACR-like violation.
        Previous implementation averaged penetration across all pairs/frames,
        which diluted rare but important ego-object collisions. This version
        uses ego-vs-agent top-k/max aggregation with early-time weights.
        """
        device = traj_ego.device
        B, T, _ = traj_ego.shape
        Np = traj_peds.shape[1]
        Nv = traj_vehs.shape[1]
        if ped_mask is None:
            ped_mask = torch.ones((B, Np), dtype=torch.bool, device=device)
        if veh_mask is None:
            veh_mask = torch.ones((B, Nv), dtype=torch.bool, device=device)

        weighted_risk = torch.zeros(B, device=device, dtype=traj_ego.dtype)
        weight_sum = torch.zeros(B, device=device, dtype=traj_ego.dtype)

        for frame_i, ts in enumerate(self.check_frames):
            if ts >= T:
                continue
            ego_t = traj_ego[:, ts, :4]
            ped_t = traj_peds[:, :, ts, :4]
            veh_t = traj_vehs[:, :, ts, :4]
            pen_ped = self._pair_ego_vs_group_dense(ego_t, ped_t, ped_mask, self.ped_shape)
            pen_veh = self._pair_ego_vs_group_dense(ego_t, veh_t, veh_mask, self.veh_shape)
            ego_agent_pen = torch.cat([pen_ped, pen_veh], dim=1)
            frame_risk = self._aggregate_topk(ego_agent_pen)
            if self.veh_veh_weight > 0:
                frame_risk = frame_risk + self.veh_veh_weight * self._pair_veh_vs_veh(veh_t, veh_mask)
            w = self.time_weights[min(frame_i, len(self.time_weights) - 1)] if self.time_weights else 1.0
            weighted_risk = weighted_risk + float(w) * frame_risk
            weight_sum = weight_sum + float(w)

        return weighted_risk / weight_sum.clamp_min(1e-6)

    def forward(self, traj_ego, traj_peds, traj_vehs, ped_mask, veh_mask):
        g = self.compute_constraint_violation(traj_ego, traj_peds, traj_vehs, ped_mask, veh_mask)
        return (self.lambda_val * g).mean()
