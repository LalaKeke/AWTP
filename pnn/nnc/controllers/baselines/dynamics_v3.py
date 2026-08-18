from numbers import Number
from typing import Union
from typing import Iterable

import torch
from nnc.controllers.base import ControlledDynamics
## 离散纵向车辆模型
class DiscreteTimeInvariantDynamics(ControlledDynamics):
    def __init__(self,
                 a,
                 b,
                 dtype=torch.float32,
                 device=None
                 ):
        super().__init__(['x'])
        self.device = device or ("cuda:" + str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.A = a.unsqueeze(0).to(self.device)
        self.B = b.unsqueeze(0).to(self.device)
        self.label = 'linear'

    def forward(self, t: Union[torch.Tensor, float],
                x: torch.Tensor,
                u: torch.Tensor,
                d: torch.Tensor = None
                ) -> torch.Tensor:
        dx = torch.matmul(self.A, x.unsqueeze(-1)).squeeze(-1)
        if u is not None:
            control_term = torch.matmul(self.B, u.unsqueeze(-1)).squeeze(-1)
            dx += control_term
        return dx
class ContinuousTimeInvariantDynamics(ControlledDynamics):
    def __init__(self,
                 interaction__matrix,
                 driver_matrix,
                 disturbance_matrix,
                 Veh_f,
                 Veh_M,
                 Veh_g,
                 dtype=torch.float32,
                 device=None
                 ):
        """
        Coontinuous time time-invariant linear dynamics of the form: `dx/dt = Ax + Bu`.
        :param interaction__matrix: The interaction matrix `A`, that determines how states of `n_nodes`
        nodes interact, shape `n_nodes x n_nodes`.
        :param driver_matrix: The driver matrix B, which determines how `k` control signals are
        applied in the linear dynamics, shape `n_nodes x k`.
        :param dtype: torch datatype of the calculations
        :param device: torch device of the calculations, usually "cpu" or "cuda:0"
        """
        super().__init__(['x'])
        self.device = device
        self.dtype = dtype
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:" + str(torch.cuda.current_device())
            else:
                self.device = torch.device("cpu")
        # un-squeeze matrices in the first dimension so that operation broadcasts across batches[
        self.interaction__matrix = interaction__matrix.unsqueeze(0).to(device)
        self.disturbance_matrix = disturbance_matrix.unsqueeze(0).to(device)
        self.driver_matrix = driver_matrix.unsqueeze(0).to(device)
        self.Veh_f=Veh_f
        self.Veh_M=Veh_M
        self.Veh_g=Veh_g
        self.label = 'linear'

    def forward(self, t: Union[(torch.Tensor, Number)],
                x: Union[torch.Tensor, Iterable[torch.Tensor]],
                u: torch.Tensor = None,
                d: torch.Tensor = None,
                ):
        """
        Evaluation of the derivative or **amount of change** for controlled continuous-time
        time-invariant linear dynamics.
        :param x: current state values for nodes. Please ensure the input is not permuted, unless you know
        what you doing.
        :param t: time scalar, which is not used as the model is time invariant.
        :param u: control vectors. In this case please confirm  it has proper dimensionality such
        that
        torch.matmul(driver_matrix, u) is possible.
        :return: the derivative tensor.
        """
        if not isinstance(x, torch.Tensor) and isinstance(x, Iterable):
            # in case somehow state variable batches passed as a tuple or list.
            x = torch.stack(list(x))

        # batch matrix multiplication, broadcasting at dimension 0 (batch dimension)
        # a for loop and normal matrix multiplication can be used instead.

        # calculate  matrix product `<A,x>`
        dx = torch.matmul(self.interaction__matrix, x.unsqueeze(-1)).squeeze(-1)
        dx= dx.unsqueeze(0)
        # print("t_dynamics = ", t)

        if u is not None:
            # if control signals are provided, calculate matrix product `<B,u>`
            control_term = torch.matmul(self.driver_matrix, u.unsqueeze(-1)).squeeze(-1)
            # add both parts
            dx += control_term
        if d is not None:
            du = self.Veh_f*self.Veh_M*self.Veh_g*torch.cos(d) + self.Veh_M*self.Veh_g*torch.sin(d)
            du.to(self.device)
            # if disturbance signals are provided, calculate matrix product `<Bd,d>`
            disturbance_term = torch.matmul(self.disturbance_matrix, du.unsqueeze(-1)).squeeze(-1)
#             print("disturbance_term= ", disturbance_term)
            # add both parts
            dx += disturbance_term
#         print("t = ", t)
#         print("d = ", d)
        return dx

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
        """
        单车模型的前向传播，计算状态变化量 dx
        :param _: 时间（未使用）
        :param x: 当前状态 [x_pos, y_pos, theta, v]
        :param u: 控制输入 [a, delta]
        :param d: 干扰项（未使用）
        :return: 状态变化量 dx [dx_pos, dy_pos, dtheta, dv]
        """
        # 确保输入形状符合要求
        if x.shape[1] != 4:
            raise ValueError("输入 x 应该包含 4 个状态变量 [x_pos, y_pos, theta, v]")
        if u.shape[1] != 2:
            raise ValueError("输入 u 应该包含 2 个控制输入 [a, delta]")

        # 解包当前状态和控制输入
        x_pos, y_pos, theta, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        # print('x:=', x_pos, y_pos, theta, v)
        a, delta = u[:, 0], u[:, 1]
        # print('u:=', a, delta)

        # 计算状态变化量 dx
        # dx_pos = (v * torch.cos(theta))
        # dy_pos = (v * torch.sin(theta))
        dx_pos = (v.clone() * torch.cos(theta.clone()))
        dy_pos = (v.clone() * torch.sin(theta.clone()))
        dtheta = (v / self.wheelbase) * torch.tan(delta)
        dv = a
        # print('dx:', dx_pos, dy_pos, dtheta, dv)
        # print('dx_pos',dx_pos.shape,'dy_pos',dy_pos.shape, 'dtheta',dtheta.shape, 'dv',dv.shape)
        # 将状态变化量堆叠成一个张量
        dx = torch.stack((dx_pos, dy_pos, dtheta, dv), dim=1)
        return dx