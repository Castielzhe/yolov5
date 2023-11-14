import time

from models.common import *


def t0():
    c3 = C3(c1=64, c2=64, n=2, shortcut=True, g=1, e=0.5)  # c3结构一般输入输出通道不变
    _x = torch.rand(4, 64, 200, 200)
    _r = c3(_x)
    print(_r.shape)
    print(c3)

def t1():
    # 池化的计算量 h*w*H*W*C
    # spp中池化的计算量 5*5*H*W*C + 9*9*H*W*C + 13*13*H*W*C = 275*H*W*C
    # sppf中池化的计算量 5*5*H*W*C + 5*5*H*W*C + 5*5*H*W*C = 75*H*W*C

    sppf = SPPF(64, 64)
    spp = SPP(64, 64)
    _x = torch.rand(4, 64, 200, 200)
    _r = sppf(_x)
    print(_r.shape)
    print(sppf)
    print('=' * 100)
    n = 50
    _t1 = time.time()
    for i in range(n):
        spp(_x)
    _t2 = time.time()
    for i in range(n):
        sppf(_x)
    _t3 = time.time()
    print(f'SPPF:{_t3 - _t2} SPP:{_t2 - _t1}')


def t2():
    # from torchvision import models
    # models.detection.ssd300_vgg16()
    m = Normalize(c1=512)
    _x = torch.rand(4, 512, 200, 200)
    _r = m(_x)
    print(_r.shape)
    print(_x[0, 0, :10, :10])
    print(_r[0, 0, :10, :10])


def t3():
    down = DownSampling(3, 64)
    up = UpSampling(64, 128)
    _x = torch.rand(4, 3, 608, 608)
    _r = down(_x)
    print(_r.shape)



if __name__ == '__main__':
    # t0()
    # t1()
    # t2()
    t3()