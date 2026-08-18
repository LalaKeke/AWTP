import torch

def OdeEuler_solver(y0,t,dy):
    # print("y0=", y0)
    # print("t=", t)
    # print("dy=", dy)
    y = torch.zeros_like(t)  # 创建与时间点张量相同形状的全零张量
    y[0] = y0  # 设置初始条件

    dt = (t[-1] - t[0])/len(t)  # 计算时间步长


    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * dy[i-1] # 使用 Euler 方法更新数值解

    return y