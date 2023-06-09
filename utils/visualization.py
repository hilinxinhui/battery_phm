"""
1. 模型结构可视化
2. 训练过程可视化
3. 模型评估可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["font.family"] = ["SimHei"]  # 设置字体家族
# plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置无衬线字体

plt.rcParams['font.serif'] = ['STSong']

def plot_single(data, names, title):
    plt.figure(figsize=(5, 4))
    for i in range(len(data)):
        plt.plot(i)
    plt.legend(names)
    plt.title(title)
    plt.show()

def save_diagram(index, data, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(index, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")
    plt.clf()

# 绘制 1 * 4 容量退化曲线
def cap_viz(data, names, title):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(title)
    for i in range(4):
        axes[i].plot(data[i])
        axes[i].set_title(names[i])
        axes[i].set_xlabel("循环圈数")
        axes[i].set_ylabel("放电容量（Ah）")
        axes[i].annotate(
            f"({chr(97+i)})",
            xy=(0.5, -0.25),
            fontsize=12,
            ha="center",
            va="center",
            xycoords="axes fraction",
        )
    plt.tight_layout()
    plt.plot()

if __name__ == "__main__":
    # plot_single 函数测试
    plot_single(list(range(10)), ["demo_curve"], "demo")
