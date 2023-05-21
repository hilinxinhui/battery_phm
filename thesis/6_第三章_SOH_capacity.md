## 3.1 引言

在锂离子电池健康状态估计和剩余寿命预测的数据驱动方法早期研究中，研究人员通常使用单一健康因子表征电池退化状态，数据驱动模型通过健康因子的历史变化数据预测其下一时间步的数值并重新映射回电池健康状态，从而将锂离子电池健康估计问题转换为一般性的时间序列预测问题。深度学习中的一个分支，循环神经网络（Recurrent Neural Network，RNN）尤其是其变体长短期记忆神经网络（Long Short-Term Memory，LSTM），可以用于处理具有时间相关性的数据；通过简单的预处理将序列数据变换为有监督样本的形式，更多深度学习模型如卷积神经网络（Convolutional Neural Network，CNN）等方法也可以用于时间序列分析和预测。同时，早期时间序列回归方法如自回归（Autoregression，AR）算法，部分机器学习方法如支持向量回归（Support Vector Regression，SVR）算法和浅层深度学习方法如多层感知机（Multi-Layer Perceptron，MLP）算法等常被用于锂离子电池健康状态估计问题中。

本章在仔细调研文献和详尽分析锂离子电池公开数据集的基础上，采用NASA数据集（B0005、B0006、B0007和B0018电池）和CALCE数据集（CS2_35、CS2_36、CS2_37和CS2_38电池），使用CNN和LSTM模型开展基于历史容量数据的锂离子电池健康状态估计方法研究并横向对比AR模型、SVR模型和MLP模型，最后通过时间序列预测问题的常用评价指标分析模型性能。

## 3.2 基于长短期记忆神经网络的电池健康状态直接估计方法

### 3.2.1 长短期记忆神经网络原理

以深度学习的视角，锂离子电池SOH估计问题可以看做时间序列预测（Time Series Forecast）问题，更具体地，采用历史容量退化数据进行SOH估计的问题属于单变量有监督时间序列回归（Supervised Univariate Time Series Regression）问题。总地来说，时间序列回归问题的研究对象是形如 $\mathbf{x}  = \left \{ x_{1}, x_{2}, \ldots ,x_{i}, \ldots, x_{n} \right \} $ 的序列形式数据。其中 $i$ 称为时间步（time step），是时间索引的抽象表达。要预测在某个时间步 $t$ 的观测数据 $x_{t}$，一般先求其分布，如【式3-1】。

$$ x_{t} \sim P \left ( x_{t} | x_{t-1}, x_{t-2}, \ldots,x_{1} \right ) \tag{3-1} $$

为估计【式3-1】中表述的分布的展开，通常使用以下两种策略。第一种策略认为不必采用可能很长的时间序列 $\left \{ x_{t-1}, x_{t-2}, \ldots, x_{1} \right \}$，只采用其中一个长为 $\tau$ 的子列 $\left \{ x_{t-1}, x_{t-2}, \ldots, x_{t-\tau} \right \} \left ( t > \tau \right )$，从而使得每次处理的为一个定长序列，这种策略将子列外的数据全部舍弃，与隐状态无关，称为无隐状态模型.第二种策略采取的方法是对于过去所有观测结果，只保留对其的“总结”$h_{t}$，同时在更新预测值 $\hat{x_{t}}$ 的同时更新 $h_{t}$，如【式3-2】至【式3-4】，在这种策略中，$h_{t}$ 事实上从未被观测到，从而这种模型被称为隐状态模型。

$$ \hat{x_{t}} = P\left ( x_{t} | h_{t-1} \right ) \tag{3-2} $$
$$ h_{t} = g\left ( h_{t-1}, x_{t-1} \right ) \tag{3-3} $$
$$ P\left ( x_{t} | x_{t-1}, x_{t-2}, \ldots, x_1 \right ) \approx P\left ( x_{t} | h_{t-1} \right ) \tag{3-4} $$

一类称为RNN的网络具有具有循环连接结构，是隐状态模型的代表，其诞生正是为了解决时间序列预测问题。LSTM是RNN的一种，以下简要介绍LSTM网络原理与结构，要介绍LSTM网络，首先介绍RNN网络。RNN的结构以MLP为基础。如【图3-1(a)】为MLP结构图及其展开形式，【式3-5】和【式3-6】描述了输入层数据经过隐藏层到达输出层的过程，其中 $X$、$H$ 和 $O$ 分别为输入层向量、隐藏层向量和输出层向量，$U$ 和 $V$ 分别为输入层到隐藏层参数矩阵和隐藏层到输出层参数矩阵，$b_{1}$ 和 $b_{2}$ 分别为隐藏层偏置和输出层偏置，$\sigma$ 为激活函数。

$$ O = \sigma(VH + b_{2}) \tag{3-5} $$
$$ H = UX + b_{1} \tag{3-6} $$

RNN结构处理的问题与MLP等结构正好相反，后者假设输入样本之间独立，而前者要求输入样本之间的相关性。尽管如此，RNN的结构与MLP非常相似，在MLP基础上增加了一个闭合回路，如【图3-1(b)】为RNN结构图及其展开形式，其中 $X$、$H$ 和 $O$ 定义与MLP中相同，$b_{o}$ 和 $b_{f}$ 为隐藏层和输出层偏置，$f$ 和 $g$ 分别为隐藏层和输出层的激活函数，$W$ 为时间点之间的权重矩阵。这一闭合回路的加入，使得RNN在某一时刻生成某一输入的输出时不止仅仅依据该时刻的输入，同时依据上一时刻输入经过隐藏层后的生成结果（即上一时刻隐藏层的输出，这里的讨论以Elman结构为基础，另一种名为Jordan的结构使用上一时刻输出层的结果），如【式3-7】和【式3-8】，这里闭合回路直观上解释为“记忆单元”。

$$ O_{t} = g(VH_{t} + b_{o}) \tag{3-7}$$
$$ H_{t} = f(UX_{t} + WH_{t-1} + b_{f}) \tag{3-8}$$

<figure>
<figcaption>图3-1 多层感知机和循环神经网络结构示意图</figcaption>
<img src="../assets/thesis_figures/chapter_3/mlp_structure.jpg">
<img src="../assets/thesis_figures/chapter_3/simple_rnn_structure.jpg">
</figure>

RNN具有很多变种，区别在于使用了不同的记忆单元，以上讨论的RNN结构是最简单的RNN结构，不妨称其为最简RNN（Simplest RNN，SRNN），其记忆单元对“记忆”信息的写入和读出没有任何控制，导致SRNN模型存在一些问题。具体地，MLP模型通过反向传播时的自动微分和随机梯度下降学习参数，RNN模型的训练过程与此类似，通常称为时间反向传播（Backpropagation Through Time，BPTT），在此过程中，设输入时间序列的长度为 $n$，在参数学习过程中将计算 $n$ 个时间步的梯度，产生长度为 $O(n)$ 的矩阵乘法链。当 $n$ 很大时，这种训练策略将导致数值不稳定，这种数值不稳定性即梯度爆炸或梯度消失。梯度爆炸和梯度消失是限制SRNN模型广泛应用的主要因素。

为克服上述问题，需要对RNN中的记忆单元进行进一步的设计，对记忆信息的写入和读出做更多限制，LSTM结构是一种设计方案。由前所述，RNN网络的结构核心在于记忆单元。如【图3-2】(a)所示为LSTM网络结构中记忆单元的设计示意图，【图3-2】(b)在【图3-2】(a)基础上标明了各部分输入输出信号及各部分计算方式。LSTM在SRNN的基础上向记忆单元加入了三个控制门，分别是输入门（input gate）、输出门（output gate）和遗忘门（forget gate），三个门受输入信号控制，依据输入信号决定开闭和开放程度。SRNN结构中的记忆信息自动地从记忆单元中读出和写入，在LSTM中，输入门控制是否将记忆信息写入记忆单元，输出门控制是否将记忆信息从记忆单元中读出，遗忘门则决定当前存储在记忆单元中地来自过去地记忆信息是否还重要以决定是否将其遗忘，这样的设计使得LSTM网络不知直接利用上一个时间步的记忆信息，具有更强的记忆回溯能力。以下给出上述内容的数学表述，LSTM网络的记忆单元接受一个样本数据，记为 $X_{t}$，$X_{t}$ 经过三次线性变换，生成三个相同维度的向量，分别是 $Z$、$Z_{i}$、$Z_{o}$ 和 $Z_{f}$，分别为输入、输入门控制信号（input gate control signal）、输出门控制信号（output gate control signal）和遗忘门控制信号（forget gate control signal），用于线性变换的矩阵与MLP网络层间的参数矩阵类似，其元素均为可训练参数（trainable parameter），三个矩阵分别记为 $W_{xi}$、$W_{xo}$ 和 $W_{xf}$，该线性变换如【式3-9】至【式3-11】，其中 $\sigma$ 为激活函数。设在第 $t$ 个时间步，更新前记忆单元内记忆信息为 $C_{t-1}$，更新后记忆信息为 $C_{t}$，更新过程如【式3-12】，其中 $f$ 为某种激活函数，后文将对LSTM网络的输入进行修正，修正前暂时将被激活的数据记为 $Z$。最后更新后的记忆信息在遗忘门信号的控制下输出，如【式3-13】。

<figure>
<figcaption>图3-2 长短期记忆神经网络结构示意图</figcaption>
<img src="../assets/thesis_figures/chapter_3/lstm_1.jpg">
<img src="../assets/thesis_figures/chapter_3/lstm_2.jpg">
<img src="../assets/thesis_figures/chapter_3/lstm_3.jpg">
</figure>

$$ I_{t} = \sigma (X_{t} W_{xi}) = \sigma (Z_{i}) \tag{3-9} $$
$$ F_{t} = \sigma (X_{t} W_{xf}) = \sigma (Z_{f}) \tag{3-10} $$
$$ O_{t} = \sigma (X_{t} W_{xo}) = \sigma (Z_{o}) \tag{3-11} $$
$$ C_{t} = F_{t} \odot C_{t-1} + I_{t} \odot f(Z) \tag{3-12} $$
$$ H_{t} = O_{t} \odot tanh(C_{t}) \tag{3-13} $$

进一步地，LSTM网络会将 $C_{t-1}$、$H_{t-1}$ 和 $X_{t}$ 一起作为模型输入，如【图3-2】(c)，对于输入 $H_{t-1}$，与 $X_{t}$ 类似，也要进行线性变换，参与门控信号的生成，这样，【式3-9】至【式3-11】修正为【式3-14】至【式3-16】，其中 $b_{i}$、$b_{f}$ 和 $b_{o}$ 为三个偏置参数。同时由于 $H_{t-1}$ 的加入，$C_{t}$ 的更新也需要修正，引入候选记忆元，其生成过程如【式3-17】，其中 $W_{xc}$ 和 $W_{hc}$ 为权重参数，$b_{c}$ 为偏置参数。记忆单元中记忆信息的更新过程如【式3-18】。隐状态的更新过程保持不变，如【式3-13】。

$$ I_{t} = \sigma (X_{t} W_{xi} + H_{t-1} W_{hi} + b_{i}) \tag{3-14} $$
$$ F_{t} = \sigma (X_{t} W_{xf} + H_{t-1} W_{xf} + b_{f}) \tag{3-15} $$
$$ O_{t} = \sigma (X_{t} W_{xo} + H_{t-1} W_{xo} + b_{o}) \tag{3-16} $$
$$ \tilde{C_{t}} = tanh(X_{t} W_{xc} + H_{t-1} W_{hc} + b_{c}) \tag{3-17} $$
$$ C_{t} = F_{t} \odot C_{t-1} + I_{t} \odot \tilde{C_{t}} \tag{3-18} $$

LSTM可以缓解梯度消失梯度爆炸问题，在长时间序列处理能力上显著优于SRNN。

### 3.2.2 长短期记忆神经网络模型

（介绍具体模型结构和超参数）

使用平均绝对误差（Mean Average Error，MAE）作为模型损失函数，其中 $n$ 为循环圈数，$\mathbf{y} = \left \{ y_{1}, y_{2}, \ldots, y_{n} \right \} $ 为容量真值，$\hat {\mathbf{y}} = \left \{ \hat{y_{1}} , \hat{y_{2}} , \ldots, \hat{y_{n}} \right \} $ 为模型预测容量值。
$$ E_{mae} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_{i} - \hat{y}_{i} \rvert, i = 1, 2, \ldots, n \tag{3-}$$

使用Adam优化器优化模型参数。

## 3.3 基于卷积神经网络的电池健康状态直接估计方法

### 3.3.1 卷积神经网络原理

尽管卷积神经网络主要被用于计算机视觉领域处理高维数据，但许多文献表明【文献】，对时间序列输入稍加变换并辅以合理设计的网络结构，CNN同样具备时间序列处理能力，以下简单介绍卷积神经网络的结构。

CNN网络结构中的隐藏层通常包括卷积层（convolution layer）、池化层（pooling layer）和全连接层（fully-connected layer）。如【图】是一个省略了输入层和输出层的CNN网络结构示意图。CNN通过引入卷积操作替换直接进行的线性变换实现相邻层间的映射，使隐藏层中每一层上的神经元变得相对稀疏，具体地，设置一个相对小的卷积核（kernel），通过滑动窗口方法比那里输入层或中间特征图，考虑二维情形，设 $x$ 为输入，$C(a, t)$ 为卷积后得到的特征图在第 $a$ 行、第 $t$ 列 的元素值，卷积核大小为 $(m, n)$，卷积时滑动窗口的移动步幅（stride）为 $(s, d)$，权重参数为 $\omega$，偏置参数为 $b$，激活函数为 $f$，卷积过程的数学表述如【式】。对于激活函数，通常使用线性整流（Rectified Linear Unit，ReLU）函数，其定义如【式】。

<figure>
<figcaption>卷积神经网络结构示意图</figcaption>
<img src="/home/lxh/battery_phm/assets/thesis_figures/chapter_3/cnn_structure.svg">
</figure>

$$ C(a, t) = f \left ( \sum_{i=0}^{m} \sum_{j=0}^{n} x (a \times s+i, t \times d + j) \omega (i, j) + b \right ) \tag{} $$

$$ f(x) = \max(0, x) $$

多数CNN具有卷积-池化的交错结构，池化层的作用为降采样，即降低输出到下一层的特征图维度以减小计算开销，提高计算速度。通常采用最大池化或平均池化，这里以最大池化为例，其数学表述如【】，其他符号约定与【式】中相同。

$$ P(a, t) = \mathop{\max}_{0 \le i \le m-1 \atop 0 \le j \le n-1} \left \{ C(a \times s + i, t \times d + j) \right \} \tag{} $$

全连接层和MLP一致，提供从输入到输出的映射，具体地，输入经过若干卷积-池化交替结构后输出一个包含需要的抽象特征信息的特征图，特征图在送入全连接层之前先被展平为一维向量，此时特征相应区域在展平的向量中的分布并不确定，全连接层能够消除分布差异对最终结果的影响，同时修改输出的形状，如对回归问题，高维特征图通过若干全连接层后逐步降维，最后输出一个标量数值。

综上，CNN使用三个策略减小计算开销提高计算效率，分别是稀疏连接、共享权重和池化结构。区别于MLP模型中相邻层神经元的全连接结构，CNN通过设置较小的滤波器尺寸实现相邻层之间只有部分神经元彼此连接，这样做一方面极大地减少了网络中需要优化的参数数量，另一方面能很大程度上降低模型的过拟合风险。此外，对于CNN中每一层神经元，其与相邻层所有神经元间的连接权重都相同，这样做同样为了减少训练和推理时的计算开销。最后，CNN网络引入池化结构，池化过程本质上是降采样过程，池化结构的引入使得特征图缩小只保留最重要的特征。

### 3.3.2 卷积神经网络模型

使用CNN处理时间序列，首先引入滑动窗口策略处理时序数据。与上文中介绍的卷积核在输入图/特征图上的滑动过程类似，设单变量时间序列为 $\mathbf{x} = \left \{ x_{1}, x_{2}, \ldots, x_{n} \right \}  $，引入时间窗口概念，其描述的是 $\mathbf(x)$ 的一个连续子串，设时间窗口长度为 $m$，意味着用前 $m$ 个时间步的数据预测下一个时间步的值。基于此，可将原始长时间序列重构为包含时间窗口内序列/输入向量和目标值/输出的二元组形式的数据集，如【式】。

$$\left \{
    \begin{array}{lr}
    (x_{1}, x_{2}, \ldots, x_{m}), x_{m+1}  \\
    (x_{2}, x_{3}, \ldots, x_{m+1}), x_{m+2}  \\
    \ldots \\
    (x_{n-m}, x_{n-m+1}, \ldots, x_{n-1}), x_{n} (n > m) 
    \end{array}
    \right. \tag{3-} $$

（介绍具体模型结构和超参数）

<figure>
<figcaption>CNN网络结构示意图</figcaption>
</figure>

## 3.4 实验结果与分析

本节展示CNN模型和LSTM在CALCE数据集和NASA数据集上进行锂离子电池健康状态估计的效果，并对比AR模型、SVR模型和MLP模型。后三者是早期研究中常用的机器学习/浅层神经网络模型。

除上文已经给出的MAE外，还使用最大误差（Max Error，MaxE）和均方根误差（Root Mean Squared Error，RMSE）评估模型性能。
评价指标定义分别为【式】（MAE）【式】（MaxE）和【式】（RMSE），其中 $n$ 为循环圈数，$\mathbf{y} = \left \{ y_{1}, y_{2}, \ldots, y_{n} \right \} $ 为容量真值，$\hat {\mathbf{y}} = \left \{ \hat{y_{1}} , \hat{y_{2}} , \ldots, \hat{y_{n}} \right \} $ 为模型预测容量值。。

$$MaxE = \max \limits_{1 \leq i \leq n} \lvert y_{i} - \hat{y}_{i} \rvert \tag{3-}$$

$$ E_{rmse} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^{2}} , i = 1, 2, \ldots, n \tag{3-}$$



<figure>
<figcaption>图3- AR模型在CALCE数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_35_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_36_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_37_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_38_ar.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- AR模型在NASA数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/nasa_B0005_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0006_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0007_ar.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0018_ar.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- SVR模型在CALCE数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_35_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_36_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_37_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_38_svr.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- SVR模型在NASA数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/nasa_B0005_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0006_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0007_svr.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0018_svr.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- MLP模型在CALCE数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_35_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_36_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_37_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_38_mlp.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- MLP模型在NASA数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/nasa_B0005_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0006_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0007_mlp.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0018_mlp.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- LSTM模型在CALCE数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_35_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_36_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_37_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_38_lstm.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- LSTM模型在NASA数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/nasa_B0005_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0006_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0007_lstm.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0018_lstm.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- CNN模型在CALCE数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_35_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_36_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_37_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/calce_CS2_38_cnn.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图3- CNN模型在NASA数据集上的预测结果</figcaption>
<img src="../assets/thesis_figures/chapter_3/nasa_B0005_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0006_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0007_cnn.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_3/nasa_B0018_cnn.jpg" width=400 height=300>
</figure>

## 3.5 本章小结

本章简要介绍了CNN和LSTM两种模型的原理，阐述了将这两种模型用于基于电池历史容量退化数据的电池健康状态估计问题时的数据预处理过程、网络结构设计和模型超参数配置。除此之外，本课题实现了另外三种可用于电池健康状态估计的模型，分别是无隐状态的自回归模型、支持向量回归模型和多层感知机模型，在CALCE数据集和NASA PCoE数据集上实验并依据最大误差、平均绝对误差和均方根误差归纳其预测性能。

在上述实验中，