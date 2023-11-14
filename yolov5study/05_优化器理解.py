import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim

from utils.general import one_cycle


def t0():
    net1 = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 4)
    )

    net2 = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

    net = nn.Sequential(
        net1,
        nn.ReLU(),
        net2
    )

    opt = optim.SGD(net1.parameters(), lr=0.01)
    opt.add_param_group({'params': net2.parameters(), 'lr': 0.2, 'momentum': 0.01, 'nesterov': True})
    lf = one_cycle()
    lr_ = lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)  # 每三次调用,将优化器的学习率更改为原来的0.1倍
    loss_fn = nn.MSELoss()

    _x = torch.rand(5, 2)
    _y = torch.rand(5, 1)

    for i in range(10):
        _py = net(_x)
        _loss = loss_fn(_y, _py)
        print(_loss)

        opt.zero_grad()
        _loss.backward()
        opt.step()

        lr_.step()  # 学习率更新
        print(f'更新后学习率:{lr_.get_last_lr()}')

if __name__ == '__main__':
    t0()




