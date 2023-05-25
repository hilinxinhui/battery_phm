<!-- 2023_05_01 -->

## 单变量健康状态估计

锂离子电池的放电容量不可测量，应用上通常手动构建健康因子，建立健康因子和健康状态的映射。

## NASA数据集的说明

NASA数据集中的电池在充放电循环测试中穿插进行阻抗测试，使得数据集中样本数量少，容量再生现象明显，其放电容量退化曲线与常见容量曲线退化模式有差异。

<!-- 2023_05_03 -->

## 为什么用深度学习方法实现电池SOH和RUL预测

传统的方法（主要是包括SVR、GPR和BRR在内的机器学习方法）所需的训练时间短、参数量少。

但深度学习方法：

1. 更符合工业需求（PEP指标最高）
2. 对数据集鲁棒性更高（即使适应不好，也可以通过迁移学习等通过很小的代价大幅提高泛化能力）
3. 不少新的计算架构/硬件部件被设计设计出来加速深度学习训练和推理

<!-- 2023_05_06 -->

## Matplotlib 中文

- SimHei 下载地址：https://us-logger1.oss-cn-beijing.aliyuncs.com/SimHei.ttf
- https://blog.csdn.net/weixin_45707277/article/details/118631442

## Matplotlib 多图注释

```Python
import matplotlib.pyplot as plt

# 创建 1x4 子图网格
fig, axs = plt.subplots(1, 4, figsize=(15, 3))

# 绘制第一个子图
axs[0].plot([0, 1, 2, 3], [1, 4, 9, 16])
axs[0].set_xlabel('X轴')
axs[0].set_ylabel('Y轴')
axs[0].set_title('(a)子图标题1')

# 绘制第二个子图
axs[1].plot([0, 1, 2, 3], [1, 2, 3, 4])
axs[1].set_xlabel('X轴')
axs[1].set_ylabel('Y轴')
axs[1].set_title('(b)子图标题2')

# 绘制第三个子图
axs[2].plot([0, 1, 2, 3], [1, 3, 5, 7])
axs[2].set_xlabel('X轴')
axs[2].set_ylabel('Y轴')
axs[2].set_title('(c)子图标题3')

# 绘制第四个子图
axs[3].plot([0, 1, 2, 3], [2, 4, 6, 8])
axs[3].set_xlabel('X轴')
axs[3].set_ylabel('Y轴')
axs[3].set_title('(d)子图标题4')

# 添加带编号的注释
for i in range(4):
    axes[i].annotate(f'({chr(97+i)})', xy=(0.5, -0.2), fontsize=12, ha='center', va='center', xycoords='axes fraction')

# 显示图形
plt.show()
```

<!-- 2023_05_09 -->

## Matplotlib 保存图片

```Python
plt.savefig()
```

- 用`fname`字段指定图片保存的后缀（常用的如jpg格式图片和eps矢量图）
- 用`dpi`字段指定图像分辨率
- 设置`bbox_inches="tight"`自适应地找到一个能包含所有元素的框
- figsize和dpi的关系：https://www.cnblogs.com/lijunjie9502/p/10327151.html#:~:text=matplotlib%20%E4%B8%AD%E8%AE%BE%E7%BD%AE%E5%9B%BE%E5%BD%A2%E5%A4%A7%E5%B0%8F%E7%9A%84%E8%AF%AD%E5%8F%A5%E5%A6%82%E4%B8%8B%EF%BC%9A%20fig%20%3D,plt.figure%20%28figsize%3D%20%28a%2C%20b%29%2C%20dpi%3Ddpi%29

## Matplotlib 文字中英混排

实现中文宋体、英文Times New Roman

- 解决方案之一：https://zhuanlan.zhihu.com/p/118601703
- 解决方案之二：https://www.zhihu.com/question/344490568/answer/936561524

<!-- 2023_05_10 -->

## 电池数据集

现阶段的电池退化周期非常长，即使只获得加速老化的电池数据也非常消耗时间且对信号收集仪器（“充放电测试盒”）的要求很高。研究通常适用开源数据集。

<!-- 2023_05_13 -->

## Markdown转pdf

- 安装pandoc
- 安装TeX（TeX Live或MacTeX）
- 选好字体（macOS可以在字体册中查看）

```Shell
pandoc input.md -o output.pdf --pdf-engine=xelatex -V CJKmainfont='STSongti-SC-Regular' -V mainfont='Times New Roman'
```

<!-- 2023_05_25 -->

## matplotlib反转x轴

```Python
import matplotlib.pyplot as plt
plt.gca().invert_xaxis() # 反转x轴
# 反转y轴使用invert_yaxis()
```