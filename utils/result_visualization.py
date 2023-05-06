import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]  # 设置字体家族
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置无衬线字体


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
