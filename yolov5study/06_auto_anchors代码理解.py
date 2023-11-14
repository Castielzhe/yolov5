import numpy as np
import yaml

from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.dataloaders import create_dataloader
import numpy as np
import matplotlib.pyplot as plt


def t0(hyp):
    train_path = r'../../datasets/coco16/images/train2017'
    cache = None  # ram/disk
    train_loader, dataset = create_dataloader(train_path,  # 文件夹路径
                                              640,  # 图像大小
                                              4,  # 批次大小
                                              32,  # 网络结构中,最大的featuremap缩放比例, 默认为32
                                              False,  # 是否是单类别
                                              hyp=hyp,  # 超参数dict字典对象
                                              augment=True,  # 是否做数据增强
                                              cache=cache,  # 数据缓存
                                              rect=False,
                                              rank=-1,
                                              workers=0,  # 数据加载的线程数量
                                              image_weights=None,
                                              quad=False,
                                              shuffle=True,
                                              )

    cfg = '../models/yolov5s_copy_auto_anchors.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )

    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=640)  # run AutoAnchor 自动计算anchor box先验框大小


def run():
    hyp = r'../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    t0(hyp)


def t1():
    def sigmoid(_x):
        return 1.0 / (1.0 + np.exp(-_x))

    def f1(_x):
        return 4 * sigmoid(_x)

    def f2(_x):
        return (2 * sigmoid(_x)) ** 2

    tw = np.arange(20, step=0.1) - 10
    y1 = list(f1(tw))
    y2 = list(f2(tw))
    print(y2)

    plt.plot(tw, y1, 'r', label='f1')
    plt.plot(tw, y2, 'b', label='f2')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    run()
