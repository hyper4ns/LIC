# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import random
import os
import re

import numpy
import torch
import torch.nn as nn
from models.layers import PackAnalysisLayer, PackSynthesisLayer

def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    text = 'ab12cd34ef56'
    reStr = re.sub("\d+", "34", text)
    print(reStr)


    # device = 'cuda'
    # ten1 = torch.ones([4, 2, 2, 2]).to(device)
    # ten2 = ten1 + ten1
    #
    # pseudo_flag = torch.ones([4, 1, 1, 1], device=device)
    # print(pseudo_flag)
    # pseudo_flag = torch.where(torch.rand([4, 1, 1, 1], device=device) < 0.5,
    #                           pseudo_flag, torch.zeros_like(pseudo_flag))
    # print(pseudo_flag)
    #
    # print(ten1 * pseudo_flag + ten2 * (1- pseudo_flag))


    # path = "D:\\Project\\LIC\\dataset\\Kodak24\\test"
    # filelist = os.listdir(path)
    # count = 0
    # size = 0
    # for file in filelist:
    #     count += 1
    #     dir = os.path.join(path, file)
    #     size += os.path.getsize(dir)
    # size /= count
    # print(size)

    # transform = PackAnalysisLayer(2)
    # inverse = PackSynthesisLayer(2)
    #
    # mask = torch.randint(10, [2, 12, 4, 4])
    # # print(mask)
    # packs = transform(mask)
    # for i, pack in enumerate(packs):
    #     a = random.random()
    #     if a < 0.5:
    #         pack[:, :, :, :] = 0
    # mask = inverse(packs)


    # print(mask)


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
