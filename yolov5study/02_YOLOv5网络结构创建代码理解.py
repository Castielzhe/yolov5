import torch.onnx
import sys


print(sys.path)
sys.path.append('D:\PythonFiles\DeepLearning\\07_YOLO\yolov5')
print(sys.path)
# noinspection PyUnresolvedReferences
from models.yolo import Model  # 实际模块位置在sys.path.models.yolo中


def t0():
    cfg = '../models/yolov5s_copy.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )
    print(model)
    r = model(torch.rand(4, 3, 608, 608))
    print(r)

    # 转换成onnx
    torch.onnx.export(
        model,
        torch.rand(4, 3, 608, 608),
        'yolov5s_copy.onnx',
        input_names=['features'],
        output_names=['labels'],
        opset_version=12,
    )


def t1():
    cfg = '../models/yolov5s_copy_01.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )
    print(model)
    r = model(torch.rand(4, 3, 608, 608))
    print(r)

    # 转换成onnx
    torch.onnx.export(
        model,
        torch.rand(4, 3, 608, 608),
        'yolov5s_copy_01.onnx',
        input_names=['features'],
        output_names=['labels'],
        opset_version=12,
    )


def t2():
    # 检测边框大小的更改
    cfg = '../models/yolov5s_copy_02.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )
    print(model)
    r = model(torch.rand(4, 3, 608, 608))
    print(r)

    # 转换成onnx
    torch.onnx.export(
        model,
        torch.rand(4, 3, 608, 608),
        'yolov5s_copy_02.onnx',
        input_names=['features'],
        output_names=['labels'],
        opset_version=12,
    )


def t3():
    # 自定义上采样 下采样
    cfg = '../models/yolov5s_copy_03.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )
    print(model)
    r = model(torch.rand(4, 3, 608, 608))
    print(r)

    # 转换成onnx
    torch.onnx.export(
        model,
        torch.rand(4, 3, 608, 608),
        'yolov5s_copy_03.onnx',
        input_names=['features'],
        output_names=['labels'],
        opset_version=12,
    )


def t4():
    # 检测边框大小的更改
    cfg = '../models/yolov5s_copy_04.yaml'
    model = Model(
        cfg,  # 模型结构配置文件路径信息
        ch=3,  # 输入通道数量
        nc=None,  # 给定类别数目
        anchors=None  # 给定anchor先验框大小
    )
    print(model)
    # 训练的时候,数据返回的是各个分支的原始数据(80+1)个置信度 (4)个回归系数 -->类型是:list[tensor], list中的每个tensor类型均为:[N, na, H, W, nc+1+4]
    r = model(torch.rand(4, 3, 608, 608))
    print(type(r))
    for rr in r:
        print(rr.shape)

    model.eval()
    # 默认情况下,推理返回的是一个二元组
    # 二元组的第一个元素是推理预测的最终结果, tensor对象, shape为:[N, ?, (4+1+nc)] ?表示每个图像最终预测的边框数量 4->xywh中心点宽度高度 1->是否有物体的概率 nc->有各个类别物体的概率
    # 二元组的第二个元素实际上就是训练结构中forward返回的list[tensor]列表(tensor结构是interface推理的结果)
    r = model(torch.rand(4, 3, 608, 608))

if __name__ == "__main__":
    # t0()
    # t1()
    # t2()
    # t3()
    t4()