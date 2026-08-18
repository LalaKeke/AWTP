from typing import Union
import torch
from nnc.controllers.base import ControlledDynamics

class BicycleModel(ControlledDynamics):
    def __init__(self, wheelbase: float, dt: float = 0.1):
        super().__init__(['x_pos', 'y_pos', 'theta', 'v'])

        if torch.is_tensor(wheelbase):
            wb = wheelbase.detach().clone().float()
        else:
            wb = torch.tensor(float(wheelbase), dtype=torch.float32)

        self.register_buffer("wheelbase", wb)
        self.dt = dt

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        t: Union[torch.Tensor, float] = None,
        d: torch.Tensor = None
    ) -> torch.Tensor:
        wheelbase = self.wheelbase.to(device=x.device, dtype=x.dtype)

        theta = x[:, 2]
        v = x[:, 3]
        a = u[:, 0]
        delta = u[:, 1]

        theta_normalized = torch.atan2(torch.sin(theta), torch.cos(theta))
        v_safe = torch.clamp(v, min=1e-8)

        dx_pos = v_safe * torch.cos(theta_normalized)
        dy_pos = v_safe * torch.sin(theta_normalized)
        dtheta = (v_safe / wheelbase) * torch.tan(delta)
        dv = a

        dx = torch.stack((dx_pos, dy_pos, dtheta, dv), dim=1)
        return dx