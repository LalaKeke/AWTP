import torch
import numpy as np
from nnc.controllers.base import ControlledDynamics, BaseController

# Vehicle_M=torch.tensor([1770+65*4])
# Vehicle_g=torch.tensor([9.81])
# Vehicle_f=torch.tensor([0.0083])
# N_step = 21
# N_disturbance=2

#坡度数据
# random_doubles = np.random.uniform(low=-2, high=2, size=(N_step-1)*N_disturbance).astype(np.float32)
# radian=np.radians(random_doubles[:])
# d = -(Vehicle_f*Vehicle_M*Vehicle_g*np.cos(radian) + Vehicle_M*Vehicle_g*np.sin(radian)) 

# #stepstize in time domain
# delta_t = 0.4

# # 构造目标车速数据集
# v = torch.linspace(10, 120, 12)/3.6
# v_target=torch.zeros(v.size()[0], N_step-1)

# for i in range(v_target.size(0)):
#     v_target[i]=v[i]*torch.ones(N_step-1)

# # 初始化
# # XX=torch.zeros(v_target.size(0),v_target.size(1))

# # for i in range(v_target.size(0)):
# #     for j in range(v_target.size(1)):
# #         for k in range( N_disturbance*i, N_disturbance+ N_disturbance*i):
# #             XX[k][j]=0.5*Vehicle_M*v_target[i][j]**2
# # for i in range(v_target.size(0)):
# #     for j in range(i*N_disturbance*(N_step-1), (i+1)*N_disturbance*(N_step-1)):
# #         x_target[j]=
# x_target=torch.zeros(v_target.size(0)*v_target.size(1)*N_disturbance)

# num=0
# for i in range(v_target.size(0)):
#     for k in range(N_disturbance):
#         for j in range(N_step-1):
#             x_target[num]=v_target[i][j]
#             num+=1
# print("x_target=", x_target)
            
class NeuralNetworkController(BaseController):

    def __init__(self, neural_net: torch.nn.Module):
        """
        Neural network wrapper for NNC.
        Provide the neural network as a submodule.
        """
        super().__init__()
        self.neural_net = neural_net

    def forward(self, x_pos_v, y_pos_v, theta_v, v_v,
                x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
                x_pos_p, y_pos_p, theta_p, v_p,
                x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref) -> torch.Tensor:
        """
        Wrapper method for the neural network.
        It is important that time and state tensors are provided to the neural network,
        and have the required dimensionality and values for control.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: A tensor containing control values, shape: `[b, ?, ?]`
        """
        return self.neural_net(x_pos_v, y_pos_v, theta_v, v_v,
                x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
                x_pos_p, y_pos_p, theta_p, v_p,
                x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref)


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

    def forward(self, x_pos_v, y_pos_v, theta_v, v_v,
                x_pos_v_ref, y_pos_v_ref, theta_v_ref, v_v_ref,
                x_pos_p, y_pos_p, theta_p, v_p,
                x_pos_p_ref, y_pos_p_ref, theta_p_ref, v_p_ref):
        """
        Calculates the derivative or **amount of change** under neural network control for the
        given dynamics.
        Preserves gradient flows for training.
        :param t: A tensor containing time values, shape: `[b, 1]` or `[1]` for shared time
        across batch.
        :param x: A tensor containing state values, shape: `[b, m, n_nodes ]`
        :return: `dx` A tensor containing the derivative (**amount of change**) of `x`,
        shape: `[b, m, n_nodes]`
        """
#         global X_target
        #slope data
#         print("t = ",t)
        # x_target=self.get_next_target()
        # d=self.get_next_d()
        u = self.nnc(x_pos_v = x_pos_v, y_pos_v = y_pos_v, theta_v = theta_v, v_v = v_v,
                     x_pos_v_ref = x_pos_v_ref, y_pos_v_ref = y_pos_v_ref, theta_v_ref = theta_v_ref, v_v_ref = v_v_ref,
                     x_pos_p = x_pos_p, y_pos_p = y_pos_p, theta_p = theta_p, v_p =v_p,
                     x_pos_p_ref = x_pos_p_ref, y_pos_p_ref = y_pos_p_ref, theta_p_ref = theta_p_ref, v_p_ref = v_p_ref
                     )
        #print("t = ", t)
#         print("d = ", d)
        # dx = self.underlying_dynamics(t=t, u=u, x=x, d=d)
        
        return u
    
#     def get_next_d(self):
#         d_value=d[self.index]
# #         print("d_value = ",d_value)
#         self.index += 1  # 增加索引
#         if self.index >= len(d):  # 如果索引超过d的长度
#             self.index = 0  # 重置索引为0，重新开始新一轮的输入过程
#         return d_value
    
    # def get_next_target(self):
    #     target_value=x_target[self.index_target]
    #     self.index_target += 1  # 增加索引
    #     if self.index_target >= len(x_target):  # 如果索引超过d的长度
    #         self.index_target = 0  # 重置索引为0，重新开始新一轮的输入过程
    #     return target_value
    
