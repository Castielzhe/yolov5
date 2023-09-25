import torch


def t0():
    # torch.hub torch里的一个模块.下载github里源码直接执行(执行的是对应项目里的hubconf.py文件)
    # Model
    model = torch.hub.load(repo_or_dir=r"..",  # 给定github或本地路径
                           model="yolov5s",
                           source="local")  # 模型文件or yolov5n - yolov5x6, custom

    # Images
    img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    img = r"E:\图片\e7e0c66f768f063d213c9fab9966df2.jpg"

    # Inference 推理预测 前处理 + 模型预测 + 后处理
    results = model(img)

    # Results 结果展示
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()


if __name__ == "__main__":
    t0()
