## 5.1 引言

对于新能源汽车，整车的退役时刻和其中电池组的退役时刻未必重合，汽车动力电池回收和梯次利用有着越来越广阔的市场。准确预测电池的剩余寿命，对电池在下游市场中的应用有重要意义。

如本文绪论所述，电池寿剩余寿命由电池健康状态定义，而目前通常使用的电池健康状态由电池放电容量定义。第三章和第四章集中讨论电池健康状态/电池放电容量估计问题。本章在第三章和第四章的基础上讨论电池剩余寿命预测问题，介绍了常用的以循环圈数定义的电池剩余寿命及其应用局限性，并据此介绍以容量定义的电池剩余寿命，称为安时-剩余寿命（Ah-RUL）。进一步地，基于LSTM网络基本结构构建深度LSTM电池Ah-RUL预测模型，在Unibo Powertools数据集上训练并测试。

## 5.2 剩余寿命定义

电池剩余寿命（Remaining Useful Life，RUL）的定义已在本文绪论中介绍，彼时介绍的RUL基于电池充放电循环计数，为了与下文中讨论的另一种剩余寿命定义区分，称上述定义为电池的循环剩余寿命（cycle-RUL）。为了获取电池的循环剩余寿命，文献普遍采用以下两种策略。其一基于电池全生命周期放电容量序列，通过给定失效阈值，可知电池在某个循环达到寿命终止状态（End of Life，EOL），设在第 $n_{EOL}$ 循环电池失效，当前循环为 $n_{current}$，即为即可得循环剩余寿命定义如【式5-1】。

$$ RUL_{cycle} = n_{EOL} - n_{current} \tag{5-1} $$

不妨将这种策略称为后处理（post-process）策略。另一在数据预处理步骤完成剩余寿命生成，这种生成方式以电池真实容量退化序列取代上述过程中的预测容量退化序列，生成电池全生命周期每一次充放电循环对应的剩余寿命，将此寿命作为样本标签送入数据驱动模型，模型输出即为剩余寿命预测值,类似地，称这种处理策略为前处理（pre-process）策略。注意到后处理策略要求完整的预测容量退化序列，若采用本文第三章使用的时间序列回归模型，只需前若干循环数据即可生成所需序列，但若采用本文第四章使用的基于充放电过程直接观测数据的预测模型，在充放电过程实施前无法获得对应的循环的容量预测信息，无法生成所需序列。考虑到基于直接观测量的预测模型更具现实意义，本章讨论的剩余寿命预测方法采用前处理策略。

剩余寿命是衡量电池从当前循环开始到寿命终止状态过程中放电能力的指标。上述循环剩余寿命以剩余循环圈数作为放电能力的度量，这种度量仅当电池充放电循环遵循满充满放条件（即一个充放电循环中电池SOC从0%上升到100%，再从100%下降到0%）时才是准确的，然而在实际使用场景中，几乎没有电池的充放电过程按照以上描述进行，从而循环剩余寿命的应用价值有限。为了解决这一问题，本章介绍一种剩余寿命新定义，直接以电池在达到寿命终止状态前能释放的电量作为度量，称为电池安时剩余寿命（Ampere-hour Remaining Useful Life，Ah-RUL），直观地，安时剩余寿命对剩余的每次循环中的放电容量求和。其数学表述如【式5-2】。

$$ RUL_{Ah} = \frac{trapz(M_{current}[:, :], M_{time}[:, :]) - trapz(M_{current}[n:EOL, :], M_{time}[n:EOL, :])}{C_{norm}}  \tag{5-2} $$

其中 $M_{current}$ 表示电池充放电循环中的电流矩阵，每一行记录了一次循环的电流数据，列则按照时间展开，一个元素表示在某个循环中对应某一采样点的电流采样记录。 $M_{time}$ 为时间矩阵，与电流矩阵对应，记录了电流矩阵中每一个具体电流强度的作用时间。$[:, :]$ 记号用于指定矩阵的行列索引；逗号前为行索引，逗号后为列索引；冒号用于指定行/列索引的具体取值范围，如 $a:b$ 指从第 $a$ 个索引开始，取 $b - a$ 长度的元素，当 $a$ 和 $b$ 均不指派数值时表示从第一个元素取到最后一个元素；矩阵索引从1开始。$trapz$ 记号表示梯形法（trapzoidal method），用于积分数值计算，这里尽管其作用对象为矩阵，但本质上进行的是电流在时间上的累积过程，实际计算步骤为将矩阵对应位置元素相乘再相加。$n$ 为当前循环圈数，$EOL$ 表示电池寿命终止状态对应的循环圈数，$C_{norm}$ 为电池的额定容量，通过【式5-2】计算的是电池的归一化安时剩余寿命（Normalized Ah-RUL）。

## 5.3 基于长短期记忆神经网络的锂离子电池剩余寿命估计方法

首先说明模型的数据输入。使用UNIBO Powertools数据集，该数据集中的电池按照采用的充放电策略被划分为三种类型，分别为标准类型（Standard，S）、高电流类型（High Current，H）和带预处理步骤类型（Pre-conditioned，P），三种类型具体采使用的充放电策略已在第二章第四节中说明，这里不再赘述。上述类型使数据集中电池分组的最重要依据，也是本章实验中最关注的差异。电池实际的分组命名依据更多信息，其形式为XW-C.C-AABB-T，其中X表示电池制造商，W表示电池用途，C.C记录电池额定容量，AABB记录电池的出厂日期，T表示电池的实验类型。其中除电池实验类型外的其他参数不是本实验关注的充电，其具体可取值不做介绍。基于此，将本章中使用的电池数据归纳如【表5-1】，每个电池分组中取一颗电池数据作为测试集，测试集包含7颗电池的数据，剩余电池作为训练集，训练集中包含20颗电池的数据，本章实验不使用验证集，某些分组中有部分电池的观测数据有误，这些电池的数据将被舍弃，不参与实验。每颗电池所属测试集/训练集分组及其舍弃情况同样总结于【表5-1】。

<table>
    <caption>表5-1 UNIBO Powertools数据集划分</caption>
    <tr>
        <td>电池描述</td>
        <td>训练集电池编号</td>
        <td>测试集电池编号</td>
        <td>说明</td>
    </tr>
    <tr>
        <td>DM-3.0-4019-S</td>
        <td>000、001、002</td>
        <td>003</td>
        <td></td>
    </tr>
    <tr>
        <td>DM-3.0-4019-H</td>
        <td>009、010</td>
        <td>011</td>
        <td></td>
    </tr>
    <tr>
        <td>DM-3.0-4019-P</td>
        <td>014、015、016、017</td>
        <td>013</td>
        <td>047、049电池数据舍去</td>
    </tr>
    <tr>
        <td>EE-2.85-0820-S</td>
        <td>007、008、042</td>
        <td>006</td>
        <td></td>
    </tr>
    <tr>
        <td>EE-2.85-0820-H</td>
        <td>043</td>
        <td>044</td>
        <td></td>
    </tr>
    <tr>
        <td>DP-2.00-1320-S</td>
        <td>018、036、038、050、051</td>
        <td>039</td>
        <td>019电池数据舍去</td>
    </tr>
    <tr>
        <td>DM-4.00-2320-S</td>
        <td>040</td>
        <td>041</td>
    </tr>
</table>

基于第四章讨论，仍使用充放电过程中的直接观测量作为模型输入。具体地，为充放电过程中的电压、电流和电池表面温度，考虑到UNIBO Powertools数据集中充放电过程数据采样频率设置的相当高，处理时取若干秒为一个采样区间，取该采样区间中所有采样点的观测值的均值和标准差，3个观测量在1个采样区间中生成6个统计量作为特征，仍采用滑动窗口方法，将窗口长度设置为500。预测模型的输入样本的形状即为（500，6），原始序列的首端数据无法生成如此长的样本，构造样本时采用零填充（zero padding）策略补齐，输入网络后再使用掩膜层（masking layer）处理获取真实数据。

本章引入深度LSTM网络实现剩余寿命预测，基于第三章第二节介绍的LSTM网络基本结构，深度LSTM模型堆叠了若干LSTM层，将前一层输出 $Y_{t}$（或前一层隐状态 $H_{t}$，两者等价）作为下一个LSTM层的输入。本章使用的深度LSTM网络结构、每一层类型、输出形状和参数量如【表5-2】所示，表中省略了对样本批大小的记录。具体地，输入数据首先经过掩膜层（masking layer）去除为使样本具有等长形式填充的0元素，继而经过两个LSTM层，两个LSTM层分别具有128个和64个LSTM记忆单元（神经元），样本序列经过LSTM层生成的特征序列继续经过由三个全连接层构成的多层感知机降维到标量输出，该输出即为预测的电池安时剩余寿命。三个全连接层的神经元数量分别为64、32、1。网络中masking层生成的序列直接输出到第一个LSTM层，最后一个线性层的输出直接作为预测的剩余寿命结果，除这两个层外每层均使用扩展型指数线性单元（Scaled Exponential Linear Unit，SELU）函数作为激活函数，关于SELU函数，其定义如【式5-】，其中 $\lambda$ 取值约为1.05、$\alpha$ 取值约为1.67，使用SELU激活函数取代ReLU激活函数主要由于前者具有自归一化属性，能够更好避免梯度消失和梯度爆炸问题，同时具有更快的收敛速度。在相同的层上对权重使用L2正则化（L2 Regularization）策略，减小模型过拟合风险，正则项系数取0.0002。

$$ SELU(x) = \lambda \left\{
\begin{array}{lr}
x & , & x > 0, \\
\alpha e^{x} - \alpha & , & x \le 0
\end{array}
\right. \tag{5-3} $$


设置训练批大小为32，训练轮数为500。使用Adam优化器；使用Huber损失函数，其定义如【式5-4】，其中 $y$ 和 $\hat{y}$ 分别表示真值和预测值，$\delta$ 取1，Huber损失函数融合了MAE损失和MSE损失，对样本中的离群点具有更高的鲁棒性；设置学习率为0.000003。

$$ L_{Huber} = \left\{
\begin{array}{lr}
\frac{1}{2}(y - \hat{y})^2 & , & |y - \hat{y}| \leq \delta, \\
\delta|y - \hat{y}| - \frac{1}{2} \delta ^ 2 & , & |y - \hat{y}| > \delta
\end{array}
\right. \tag{5-4} $$

使用均方根误差（RMSE）和归一化均方根误差（Normalized Root Mean Square Error，NRMSE）评估模型预测性能，其中RMSE定义已在第三章中说明，NRMSE使用真值的极差对RMSE归一化，其定义如【式5-5】，其中 $\mathbf y$ 为电池安时剩余寿命。

$$ E_{NRMSE} = \frac{RMSE}{\max (\mathbf y) - \min (\mathbf y)} \tag{5-5} $$

实验结果及其分析见本章第四节。

<table>
    <caption>表5-2 用于RUL预测的DeepLSTM网络结构</caption>
    <tr>
        <td>层号</td>
        <td>层类型</td>
        <td>输出形状</td>
        <td>参数量</td>
    </tr>
    <tr>
        <td>1</td>
        <td>masking层</td>
        <td>(500, 6)</td>
        <td>0</td>
    </tr>
    <tr>
        <td>2</td>
        <td>LSTM层１</td>
        <td>(500, 128)</td>
        <td>69120</td>
    </tr>
    <tr>
        <td>3</td>
        <td>LSTM层２</td>
        <td>64</td>
        <td>49408</td>
    </tr>
    <tr>
        <td>4</td>
        <td>全连接层１</td>
        <td>64</td>
        <td>4160</td>
    </tr>
    <tr>
        <td>5</td>
        <td>全连接层２</td>
        <td>32</td>
        <td>2080</td>
    </tr>
    <tr>
        <td>6</td>
        <td>全连接层２</td>
        <td>1</td>
        <td>33</td>
    </tr>
</table>

## 5.4 实验结果分析

如【图5-1】展示了深度LSTM模型在UNIBO Powertools数据集上的安时剩余寿命预测值和真值的对比示意图，模型预测性能的RMSE和NRMSE指标总结如【表5-3】。模型在电池039上取得最优预测性能，预测结果的NRMSE为1.92%，在电池013上预测结果最差，对应NRMSE为18.1%，模型在测试集的7块电池上取得的平均NRMSE为5.61%，这一结果证明了本章提出的深度LSTM模型在解决电池安时寿命预测问题时的有效性。

<figure>
<figcaption>图5-1 DeepLSTM在UNIBO Powertools数据集上的预测结果（以循环圈数为自变量）</figcaption>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_1.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_2.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_3.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_4.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_5.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_6.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_cycle_7.jpg" width=400 height=300>
</figure>

<table>
    <caption>表5-3 DeepLSTM剩余寿命预测性能</caption>
    <tr>
        <td></td>
        <td>电池003</td>
        <td>电池011</td>
        <td>电池013</td>
        <td>电池006</td>
        <td>电池044</td>
        <td>电池039</td>
        <td>电池041</td>
        <td>均值</td>
    </tr>
    <tr>
        <td>RMSE</td>
        <td>3.855191</td>
        <td>3.102227</td>
        <td>10.773928</td>
        <td>10.122581</td>
        <td>13.871369</td>
        <td>8.883196</td>
        <td>4.498271</td>
        <td>7.872394714</td>
    </tr>
    <tr>
        <td>NRMSE</td>
        <td>0.035861</td>
        <td>0.02924</td>
        <td>0.181346</td>
        <td>0.028673</td>
        <td>0.047128</td>
        <td>0.019219</td>
        <td>0.051681</td>
        <td>0.056164</td>
    </tr>
</table>

考虑到实际电池的充放电策略严格遵从满充满放，应用中电池实际循环圈数并不容易确定。但是基于本文第四章讨论结果，容易通过电池部分充放电段的直接观测数据估计电池放电容量，从而以上使用电池循环圈数作为横坐标建立的安时剩余寿命-循环圈数寿命可以进一步借助循环圈数和电池放电容量/电池健康状态映射生成电池的安时剩余寿命-电池实际容量映射关系，如【图5-2】。从电池放电容量到电池安时剩余寿命的映射为单射，但由于部分电池采用的充放电策略导致其放电容量-循环圈数曲线呈现出总体线性递减、频繁局部容量再生的现象，如【图5-3】，使得其安时剩余寿命-放电容量曲线不单调且（容量，剩余寿命）点在循环早期分布较为密集，故【图5-2】采用散点图而非折线图以避免歧义。

<figure>
<figcaption>图5-2 DeepLSTM在UNIBO Powertools数据集上的预测结果（以放电容量为自变量）</figcaption>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_1.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_2.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_3.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_4.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_5.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_6.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_lstm_rul_soh_7.jpg" width=400 height=300>
</figure>

<figure>
<figcaption>图5-3</figcaption>
<img src="../assets/thesis_figures/chapter_5/unibo_test_cell4_capacity.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_test_cell5_capacity.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_test_cell6_capacity.jpg" width=400 height=300>
<img src="../assets/thesis_figures/chapter_5/unibo_test_cell7_capacity.jpg" width=400 height=300>
</figure>

## 5.5 本章小结

本章讨论了电池剩余寿命的定义和生成过程，介绍了一种新的基于容量的电池剩余寿命定义。进一步地，为实现这种剩余寿命的预测，提出了一种深度LSTM模型，在UNIBO Powertools数据集上实现了安时剩余寿命的预测，这一深度LSTM模型在包含7颗具有不同规格、采用不同充放电策略的电池构成的训练集上的预测结果的平均归一化均方根误差仅为5.61%，证明了将其应用与剩余寿命预测问题上的可行性。最后，本章通过电池放电容量-循环映射和上述电池安时剩余寿命-循环映射建立了依据电池放电容量确定电池剩余可用容量的方法，具有一定的应用价值。