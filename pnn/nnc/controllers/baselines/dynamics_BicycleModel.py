from typing import Union

import torch
from nnc.controllers.base import ControlledDynamics


class BicycleModel(ControlledDynamics):
    def __init__(self, wheelbase: float, dt: float = 0.1):
        """
        初始化单车模型
        :param wheelbase: 车辆轴距
        :param dt: 离散时间步长
        """
        # 初始化父类，传入状态变量列表
        super().__init__(['x_pos', 'y_pos', 'theta', 'v'])
        self.wheelbase = wheelbase
        self.dt = dt  # 离散时间步长

    def forward(self,
                x: torch.Tensor,
                u: torch.Tensor,
                t: Union[torch.Tensor, float] = None,
                d: torch.Tensor = None) -> torch.Tensor:
        # 确保输入形状符合要求
        if x.shape[1] != 4:
            raise ValueError("输入 x 应该包含 4 个状态变量 [x_pos, y_pos, theta, v]")
        if u.shape[1] != 2:
            raise ValueError("输入 u 应该包含 2 个控制输入 [a, delta]")

        # 解包当前状态和控制输入
        x_pos, y_pos, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        a, delta = u[:, 0], u[:, 1]

        # 计算状态变化量 dx
        dx_pos = v * torch.cos(theta)
        dy_pos = v * torch.sin(theta)
        dtheta = (v / self.wheelbase) * torch.tan(delta)
        dv = a

        # 将状态变化量堆叠成一个张量
        dx = torch.stack((dx_pos, dy_pos, dtheta, dv), dim=1)
        return dx