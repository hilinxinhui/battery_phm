"""
1. 模型结构可视化
2. 训练过程可视化
3. 模型评估可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# from torchinfo import summary


# def print_model(model):
#     # 直接打印torch模型
#     print(model)
#     # 使用torchinfo打印模型参数
#     summary(model, (1, 3, 32, 32)) # 1：batch_size， 3：channel，输入图片size：32 * 32
    

# # tensorboard可以记录模型每一层的feature map、权重、loss等信息，利用tensorboard实现训练过程的可视化
# from tensorboard import SummaryWriter
# log_path = "./runs"
# writer = SummaryWriter(log_path)
# writer.add_graph(model, torch.rand(1, 3, 32, 32))