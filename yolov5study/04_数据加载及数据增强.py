import torch
import yaml
import cv2 as cv

from utils.augmentations import random_perspective, letterbox
from utils.dataloaders import create_dataloader


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

    for imgs, targets, paths, _ in train_loader:
        print(imgs)  # [batch_size, 3, img_size, img_size]
        print(targets)  # [M, 6] 表示batch_size个图像中总共有M个真实边框
        print(paths)
        break

def t2():                    # Letterbox 自适应的图像缩放(最小缩放)
    img = cv.imread('../data/images/bus.jpg')
    img = cv.resize(img, (200, 400))

    img1, _, phw1 = letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32)
    img2, _, phw2 = letterbox(img, new_shape=(100, 100), color=(114, 114, 114), auto=True, scaleup=True, stride=32)
    img3, _, phw3 = letterbox(img, new_shape=(300, 300), color=(114, 114, 114), auto=True, scaleup=True, stride=32)
    img4, _, phw4 = letterbox(img, new_shape=(100, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32)

    print(phw1, phw2, phw3, phw4)
    print(img1.sahpe, img2.shape, img3.shape, img4.shape)
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('img3', img3)
    cv.imshow('img4', img4)

def t1():
    img = cv.imread('../data/images/bus.jpg')
    img = cv.resize(img, (640, 640))
    labels = torch.tensor([

    ])

    img1, labels1 = random_perspective(img,
                                       # targets=labels,
                                       segments=(),
                                       degrees=90,
                                       translate=.1,
                                       scale=.1,
                                       shear=10,
                                       perspective=0.0,
                                       border=(-160, -160))

    cv.imshow('img', img)
    cv.imshow('img2', img1)

    cv.waitKey(-1)
    cv.destroyAllWindows()


def run():
    hyp = r'../data/hyps/hyp.scratch-low.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    t0(hyp)


if __name__ == '__main__':
    run()
    # t1()
