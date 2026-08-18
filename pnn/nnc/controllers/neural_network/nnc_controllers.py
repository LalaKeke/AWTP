import torch
import numpy as np
from nnc.controllers.base import ControlledDynamics, BaseController
            
class NeuralNetworkController(BaseController):

    def __init__(self, neural_net: torch.nn.Module):
        """
        Neural network wrapper for NNC.
        Provide the neural network as a submodule.
        """
        super().__init__()
        self.neural_net = neural_net

    # 20250322之前版本
    # def forward(self, x_pos_v, y_pos_v, theta_v, v_v,
    #             x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
    #             x_pos_p, y_pos_p, theta_p, v_p,
    #             x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref) -> torch.Tensor:
    #     """
    #     Wrapper method for the neural network.
    #     It is important that time and state tensors are provided to the neural network,
    #     and have the required dimensionality and values for control.
    #     :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
    #     across batch.
    #     :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
    #     :return: A tensor containing control values, shape: `[b, ?, ?]`
    #     """
    #     return self.neural_net(x_pos_v, y_pos_v, theta_v, v_v,
    #             x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
    #             x_pos_p, y_pos_p, theta_p, v_p,
    #             x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref)
    # 20250403之前版本 V8
    # def forward(self, state_v: torch.Tensor, state_p: torch.Tensor) -> torch.Tensor:
    #     """
    #     Wrapper method for the neural network.
    #     It is important that time and state tensors are provided to the neural network,
    #     and have the required dimensionality and values for control.
    #     :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
    #     across batch.
    #     :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
    #     :return: A tensor containing control values, shape: `[b, ?, ?]`
    #     """
    #     return self.neural_net(state_v, state_p)
    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask, veh_mask) -> torch.Tensor:

        return self.neural_net(ego_state, ped_states, veh_states, lane_points, ped_mask, veh_mask)
class NNCDynamics(torch.nn.Module):
    def __init__(self,
                 underlying_dynamics: ControlledDynamics,
                 neural_network: torch.nn.Module,
                 ):
        """
        A constuctor that couples the controlled dynamics with the neural network.
        :param underlying_dynamics: A class implementing :class:`nnc.controllers.base.ControlledDynamics`
        :param neural_network: A neural network implementing a torch module, with inputs and
        outputs described in  :method:`nnc.controllers.base.NeuralNetworkController`
        """
        super().__init__()
        # assign nnc to the wrapper, may be considered redundant but for the sake of clarity
        self.nnc = NeuralNetworkController(neural_network)
        self.underlying_dynamics = underlying_dynamics
        # for ease of use, so that one can access the same pointer faster
        self.state_var_list = underlying_dynamics.state_var_list
        # self.index = 0
        # self.index_target = 0
#     20250322之前版本
#     def forward(self, x_pos_v, y_pos_v, theta_v, v_v,
#                 x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
#                 x_pos_p, y_pos_p, theta_p, v_p,
#                 x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref):
#         """
#         Calculates the derivative or **amount of change** under neural network control for the
#         given dynamics.
#         Preserves gradient flows for training.
#         :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
#         across batch.
#         :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
#         :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
#         shape: `[b, m, n_nodes]`
#         """
# #         global X_target
#         #slope data
# #         print("t = ",t)
#         # x_target=self.get_next_target()
#         # d=self.get_next_d()
#         u = self.nnc(x_pos_v = x_pos_v, y_pos_v = y_pos_v, theta_v = theta_v, v_v = v_v,
#                      x_pos_v_ref = x_pos_v_ref, y_pos_v_ref = y_pos_v_ref, theta_v_ref = theta_v_ref, v_v_ref = v_v_ref,
#                      x_pos_p = x_pos_p, y_pos_p = y_pos_p, theta_p = theta_p, v_p =v_p,
#                      x_pos_p_ref = x_pos_p_ref, y_pos_p_ref = y_pos_p_ref, theta_p_ref = theta_p_ref, v_p_ref = v_p_ref
#                      )
#
#         #print("t = ", t)
# #         print("d = ", d)
#         # dx = self.underlying_dynamics(t=t, u=u, x=x, d=d)
#
#         return u

    ###V8版本
    # def forward(self, state_v, state_p):
    #     """
    #     Calculates the derivative or **amount of change** under neural network control for the
    #     given dynamics.
    #     Preserves gradient flows for training.
    #     :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
    #     across batch.
    #     :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
    #     :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
    #     shape: `[b, m, n_nodes]`
    #     """

    #     u = self.nnc(state_v=state_v, state_p=state_p)

    #     # print("t = ", t)
    #     #         print("d = ", d)
    #     # dx = self.underlying_dynamics(t=t, u=u, x=x, d=d)
    #     return u
    def forward(self, ego_state, ped_states, veh_states, lane_points, ped_mask, veh_mask):

        u = self.nnc(ego_state=ego_state, ped_states=ped_states, veh_states=veh_states, lane_points=lane_points, ped_mask=ped_mask, veh_mask=veh_mask)
        return u    
        
