## 4.1 引言

上一章分析了不同模型直接基于电池容量历史数据实现电池健康状态估计的原理和算法性能。这种方法面临以下问题：其一是很难实现锂离子电池的放电容量的非侵入式测量，应用中往往手动构建与电池容量等价的健康因子，通过健康因子映射电池健康状态，但这种健康因子构建与机理模型类似，需要相对详尽的机理模型参与，同时泛化能力有限；其二，使用历史循环数据，无论是直接使用容量数据还是通过健康因子进行时间序列回归，都需要若干周期完整充放电循环数据作为输入。

以上两个问题很大程度地限制了直接使用历史容量退化数据进行电池健康状态估计方法的应用。基于此，本章提出了一种基于

## 4.2 数据预处理：时间序列-图像变换

## 4.3 基于卷积神经网络的锂离子电池健康状态间接估计方法

说明网络结构和超参数配置

## 4.4 实验结果与分析

比较

- 不用时间序列-图像变换
- 使用时间序列-图像变换

两组模型在预测性能和模型参数上的优劣

<figure>
<figcaption></figcaption>
<img src="../assets/thesis_figures/chapter_3/tri_group1_vit.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group2_vit.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group3_vit.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group4_vit.jpg" width=400 height=300>
</figure>

<figure>
<figcaption></figcaption>
<img src="../assets/thesis_figures/chapter_3/tri_group1_vit_trans.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group2_vit.jpg_trans" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group3_vit.jpg_trans" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group4_vit.jpg_trans" width=400 height=300>
</figure>

<figure>
<figcaption></figcaption>
<img src="../assets/thesis_figures/chapter_3/tri_group1_viq.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group2_viq.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group3_viq.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group4_viq.jpg" width=400 height=300>
</figure>

<figure>
<figcaption></figcaption>
<img src="../assets/thesis_figures/chapter_3/tri_group1_viq_trans.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group2_viq_trans.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group3_viq_trans.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/tri_group4_viq_trans.jpg" width=400 height=300>
</figure>

## 4.5 本章小结

（滑动窗口：）
（ts2img trans：）
（CNN结构）

本章沿用前文使用的滑动窗口方法进行锂离子电池充放电循环数据分段，直接将将分段数据作为样本，一方面实现了数据增广，满足了深度神经网络的训练过程对数据量的需求，另一方面为CNN模型对输入数据的鲁棒性奠定基础，很大程度上使得网络对任意起点（相对充放电周期而言）的输入数据都能得到准确的估计结果；其次，本章介绍了一种时间序列到图像变换方法，使得CNN模型能更好地利用输入数据的时空相关性，同时简化了CNN模型的设计；最后，本章基于上述预处理步骤，简单改进LeNet5-CNN网络结构，建立了更符合应用需求的锂离子电池健康状态估计模型，通过短时间内采集的电池充放电直接测量量（电流、电压、电池表面温度和电荷量）实现快速的精准的在线健康状态估计。