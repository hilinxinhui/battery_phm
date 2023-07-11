# Battery Prognostics and Health Management

锂离子电池（单体电池）故障检测与健康管理系统。

## 项目说明

本项目的主要内容包括：

- 数据驱动的电池健康状态（State of Health，SOH）估计
- 数据驱动的电池剩余寿命（Remaining Useful Life，RUL）预测

## 环境配置

使用 [miniconda](https://docs.conda.io/en/latest/miniconda.html) 进行环境配置和管理。

- [Pytorch环境](./rul_torch.yaml)
- [TensorFlow（Keras）环境](./rul_tf.yaml)

## 数据集

本项目使用的电池数据包括：

- NASA PCoE Batteries Dataset：https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
- CALCE Battery Dataset：https://calce.umd.edu/battery-data
- TRI Battery Dataset：https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
- UNIBO Powertools Dataset：https://data.mendeley.com/datasets/n6xg5fzsbv/1

供参考，[这篇文章](https://www.sciencedirect.com/science/article/pii/S2666546821000355)介绍了常用的锂离子电池数据集以及下载地址。

## 贡献者

<a href="https://github.com/hilinxinhui/battery_phm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hilinxinhui/battery_phm" />
</a>

<!-- 感谢 [contrib.rocks](https://contrib.rocks) 。 -->

## 许可证

[MIT](./LICENSE)