import torch
from torch import nn
import pandas as pd
import numpy as np
from torchdiffeq import odeint
# import plotly.figure_factory as ff
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple

import torch.nn.functional as F

class EluTimeControlEnhanced(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, future_steps=50):
        super().__init__()
        self.future_steps = future_steps
        self.embed_dim = embed_dim

        self.ego_encoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.ped_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.veh_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        self.map_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.attn_fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.res_fc1 = nn.Linear(embed_dim, embed_dim)
        self.res_fc2 = nn.Linear(embed_dim, embed_dim)

        self.ego_output = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2),
            nn.Tanh()
        )
        self.ped_output = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2),
            nn.Tanh()
        )
        self.veh_output = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * 2),
            nn.Tanh()
        )

    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask=None, veh_mask=None):
        B = ego_state.size(0)
        Np = ped_states.size(1)
        Nv = veh_states.size(1)

        # 编码
        ego_feat = self.ego_encoder(ego_state).unsqueeze(1)  # [B, 1, D]
        ped_feat = self.ped_encoder(ped_states)              # [B, Np, D]
        veh_feat = self.veh_encoder(veh_states)              # [B, Nv, D]
        map_feat = self.map_encoder(lane_points.view(B, -1, 2))  # [B, 40, D]

        # 拼接所有特征
        all_feat = torch.cat([ego_feat, ped_feat, veh_feat, map_feat], dim=1)  # [B, total_len, D]

        # 构建 mask
        total_len = 1 + Np + Nv + map_feat.shape[1]
        device = ego_state.device
        key_padding_mask = torch.zeros(B, total_len, dtype=torch.bool, device=device)
        if ped_mask is not None:
            key_padding_mask[:, 1:1+Np] = ped_mask  # True 表示忽略
        if veh_mask is not None:
            key_padding_mask[:, 1+Np:1+Np+Nv] = veh_mask

        # Transformer Attention
        fused_feat = self.attn_fusion(all_feat, src_key_padding_mask=key_padding_mask)

        # 残差块
        fused_feat = fused_feat + F.relu(self.res_fc2(F.relu(self.res_fc1(fused_feat))))

        # 输出
        ego_out = self.ego_output(fused_feat[:, 0])  # [B, 30*2]
        ped_out = self.ped_output(fused_feat[:, 1:1+Np])  # [B, Np, 30*2]
        veh_out = self.veh_output(fused_feat[:, 1+Np:1+Np+Nv])  # [B, Nv, 30*2]

        ego_out = ego_out.view(B, self.future_steps, 2)
        ped_out = ped_out.view(B, Np, self.future_steps, 2)
        veh_out = veh_out.view(B, Nv, self.future_steps, 2)

        return ego_out, ped_out, veh_out

def normalize(data, min_val, max_val):
    """
    将数据归一化到指定的上下限范围内，适用于不同形状的数据。
    :param data: 输入的Tensor数据
    :param min_val: 每个特征的最小值，应为一个向量
    :param max_val: 每个特征的最大值，应为一个向量
    """
    min_val = min_val.to(data.device)
    max_val = max_val.to(data.device)

    # 计算范围
    range_val = max_val - min_val
    normalized = torch.zeros_like(data)

    # 生成有效掩码，避免除以零
    valid_mask = (range_val != 0)
    if valid_mask.any():
        # 根据数据维度扩展min_val和range_val
        dims_to_expand = list(range(data.ndim - min_val.ndim))
        min_val_expanded = min_val
        range_val_expanded = range_val

        for dim in dims_to_expand:
            min_val_expanded = min_val_expanded.unsqueeze(dim)
            range_val_expanded = range_val_expanded.unsqueeze(dim)
        
        # 扩展到与数据相同的形状
        min_val_expanded = min_val_expanded.expand_as(data)
        range_val_expanded = range_val_expanded.expand_as(data)

        # 应用归一化
        normalized = (data - min_val_expanded) / range_val_expanded * 2 - 1
        # 将范围外的值设置为0
        normalized[~(range_val != 0).expand_as(data)] = 0

    return normalized

    return normalized
def normalize_ego(data, min_val, max_val):
    """
    将数据归一化到指定的上下限范围内。
    :param data: 输入的Tensor数据
    :param min_val: 每个特征的最小值，应为一个向量
    :param max_val: 每个特征的最大值，应为一个向量
    """
    # 确保min_val和max_val与data在设备上一致
    min_val = min_val.to(data.device)
    max_val = max_val.to(data.device)

    # 计算范围和偏移
    range_val = max_val - min_val
    normalized = torch.zeros_like(data)  # 初始化一个全0张量，形状与data相同

    # 处理非零范围的数据进行归一化
    valid_mask = (range_val != 0)  # 非零范围的掩码
    if valid_mask.any():
        normalized[:, valid_mask] = (data[:, valid_mask] - min_val[valid_mask]) / range_val[valid_mask] * 2 - 1

    return normalized

def normalize_vp(data, min_val, max_val):
    """
    将数据归一化到指定的上下限范围内。
    :param data: 输入的Tensor数据，形状为 [B, N, F]
    :param min_val: 每个特征的最小值，形状为 [F]
    :param max_val: 每个特征的最大值，形状为 [F]
    """
    # 确保min_val和max_val与data在设备上一致
    min_val = min_val.to(data.device)
    max_val = max_val.to(data.device)

    # 计算范围和偏移
    range_val = max_val - min_val
    normalized = torch.zeros_like(data)  # 初始化一个全0张量，形状与data相同

    # 创建有效的掩码，处理非零范围的数据进行归一化
    valid_mask = (range_val != 0)  # 非零范围的掩码
    if valid_mask.any():
        # 将min_val和range_val沿着批次和行人维度扩展
        min_val_expanded = min_val.unsqueeze(0).unsqueeze(0).expand_as(data)
        range_val_expanded = range_val.unsqueeze(0).unsqueeze(0).expand_as(data)
        valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand_as(data)

        # 仅对那些有有效范围的特征执行归一化
        normalized[valid_mask_expanded] = (data[valid_mask_expanded] - min_val_expanded[valid_mask_expanded]) / range_val_expanded[valid_mask_expanded] * 2 - 1

    return normalized
def normalize_line(data, min_val, max_val):
    """
    将数据归一化到指定的上下限范围内。
    :param data: 输入的Tensor数据，可能有不同的维度
    :param min_val: 每个特征的最小值，形状为 [F]
    :param max_val: 每个特征的最大值，形状为 [F]
    """
    min_val = min_val.to(data.device)
    max_val = max_val.to(data.device)

    # 计算范围
    range_val = max_val - min_val
    normalized = torch.zeros_like(data)

    # 生成有效掩码，避免除以零
    valid_mask = (range_val != 0)
    if valid_mask.any():
        # 沿所有维度扩展min_val和range_val以匹配data的形状
        min_val_expanded = min_val.unsqueeze(0).unsqueeze(0).expand(data.size(0), data.size(1), -1)
        range_val_expanded = range_val.unsqueeze(0).unsqueeze(0).expand(data.size(0), data.size(1), -1)
        valid_mask_expanded = valid_mask.unsqueeze(0).unsqueeze(0).expand(data.size(0), data.size(1), -1)

        # 应用归一化
        normalized[valid_mask_expanded] = (data[valid_mask_expanded] - min_val_expanded[valid_mask_expanded]) / range_val_expanded[valid_mask_expanded] * 2 - 1
        normalized[~valid_mask_expanded] = 0  # 对于无效的特征值设置为0

    return normalized
def inverse_normalize(x_norm, a, b, a_norm=-1, b_norm=1):
    """
    将归一化后的数据反归一化到指定范围
    Args:
        x_norm: 归一化后的数据，[B, future_steps, 2]
        a: 原始数据范围下界, 可为标量或形状为[1, 1, 2]
        b: 原始数据范围上界, 可为标量或形状为[1, 1, 2]
        a_norm: 归一化后的数据范围下界，默认为-1
        b_norm: 归一化后的数据范围上界，默认为1
    Returns:
        反归一化后的数据
    """
    # 确保a和b能在广播时正确应用到每个元素
    a = torch.as_tensor(a, device=x_norm.device).reshape(1, 1, -1)
    b = torch.as_tensor(b, device=x_norm.device).reshape(1, 1, -1)

    x = (x_norm - a_norm) * (b - a) / (b_norm - a_norm) + a
    return x


def todf(trajectory, lr=None):
    """
    Converts a numpy tensor of two node single state variable trajectories to a dataframe for
    easier visualization.
    :param trajectory:
    :param lr:
    :return: the dataframe with all trajectories and metadata
    """
    # 检查轨迹是否为torch.Tensor类型，如果不是，则将其转换为torch.Tensor类型，并使用detach()方法确保不会对其进行反向传播
    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory)
    else:
        trajectory = trajectory.detach()
    # 定义一个空列表df_all_trajectories，用于保存所有轨迹的DataFrame
    df_all_trajectories = []
    # 获取轨迹长度n_timesteps
    n_timesteps = trajectory.shape[-2]
    # 遍历所有轨迹
    for i in range(trajectory.shape[0]):
        # 定义样本ID
        sample_id = i
        # 定义时间步骤
        timestep_ids = np.arange(trajectory.shape[-2])
        # 定义DataFrame，包含x1、x2和u三个变量，分别表示每个节点的状态变量和控制变量，以及样本ID和时间步骤
        df_trajectory = pd.DataFrame(dict(
            x1=trajectory[sample_id, :, 0],
            # x2=trajectory[sample_id, :, 1],
            u=trajectory[sample_id, :, 1],
            sample_id=i,
            time=timestep_ids,
        ))
        # 判断轨迹是否到达了最后一个时间步骤
        df_trajectory['reached'] = df_trajectory['time'] == n_timesteps - 1
        # 如果参数lr不为None，将其添加到DataFrame中
        if lr is not None:
            df_trajectory['lr'] = lr
        # 将DataFrame添加到df_all_trajectories列表中
        df_all_trajectories.append(df_trajectory)
    # 将所有DataFrame连接起来
    df_all_trajectories = pd.concat(df_all_trajectories)
    # 返回结果
    return df_all_trajectories


def evaluate_trajectory(dynamics, controller, x0, total_time, n_timesteps, method='rk4',
                        options=None):
    # 初始化
    all_controls = []
    all_control_times = []
    all_timesteps = torch.linspace(0, total_time, n_timesteps)

    # 定义函数 apply_control，用于应用控制器，计算控制信号并返回状态变量变化率
    def apply_control(t, x):
        # 计算控制信号
        u = controller(t, x)
        all_control_times.append(t)
        all_controls.append(u)
        # 计算状态变量变化率
        dx = dynamics(t=t, x=x, u=u)
        return dx

    # 将ODE Solver应用于apply_control函数，计算轨迹
    trajectory = odeint(apply_control,
                        x0,
                        all_timesteps,
                        method=method,
                        options=options
                        )  # timesteps x n_nodes

    # 将所有控制信号和控制时间步骤存储到列表中
    all_controls = torch.stack(all_controls, 0)  # timesteps x n_nodes
    all_control_times = torch.stack(all_control_times)  # timesteps x 1

    # 将ODE Solver得到的控制时间步骤与请求的时间步骤对齐
    _, relevant_time_index = closest_previous_time(all_timesteps, all_control_times)
    relevant_controls = all_controls[relevant_time_index, :]

    # 将轨迹和控制信号连接起来并返回结果
    return torch.cat([trajectory, relevant_controls], -1)


def closest_previous_time(requested_times, solver_times):
    requested_times = requested_times.unsqueeze(1)
    solver_times = solver_times.unsqueeze(0)
    difft = (requested_times - solver_times)
    difft = difft
    difft[difft < 0] = np.infty
    time_index = difft.argmin(1).flatten()
    return solver_times.squeeze()[time_index], time_index


def compare_trajectories(linear_dynamics,
                         oc_baseline,
                         nnc,
                         x0,
                         x_target,
                         T,
                         x1_min,
                         x1_max,
                         x2_min,
                         x2_max,
                         n_points=200,
                         ):
    trajectory = evaluate_trajectory(linear_dynamics,
                                     oc_baseline,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    oc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    trajectory = evaluate_trajectory(linear_dynamics,
                                     nnc,
                                     x0,
                                     T,
                                     n_points,
                                     method='rk4',
                                     options=dict(step_size=T / n_points)
                                     )
    nnc_trajectory = todf(trajectory.squeeze(1).unsqueeze(0))
    fig_trajectories = plot_trajectory_comparison(linear_dynamics,
                                                  x0,
                                                  x_target,
                                                  nnc_trajectory,
                                                  oc_trajectory,
                                                  x1_min,
                                                  x1_max,
                                                  x2_min,
                                                  x2_max
                                                  )

    energy_nnc = ((nnc_trajectory['u'] ** 2) * T / n_points).cumsum()
    energy_oc = ((oc_trajectory['u'] ** 2) * T / n_points).cumsum()
    time = nnc_trajectory.index * T / n_points

    ocen = go.Scatter(x=time, y=energy_oc, name='OC',
                      mode='lines', line=dict(color='#271f30', dash='dot'))
    nncen = go.Scatter(x=time, y=energy_nnc, name='NODEC',
                       mode='lines', line=dict(color='#ff9f00', dash='dot'))
    fig_energies = go.Figure([ocen, nncen])
    figs = make_subplots(1, 2)
    for trace in fig_trajectories.data:
        figs.append_trace(trace, 1, 1)
    for trace in fig_energies.data:
        trace.showlegend = False
        trace.xaxis = 'x2'
        trace.yaxis = 'y2'
        figs.append_trace(trace, 1, 2)
    figs.update_layout(base_temp)
    figs.update_xaxes(axis_temp)
    figs.update_yaxes(axis_temp)
    figs.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    figs.layout.height = 400
    return figs, fig_trajectories, fig_energies