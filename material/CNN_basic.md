## 基本概念
### 卷积
- [卷积](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
1. 作用：提取特征
2. 超参数:改变输出特征图的大小

```
padding（填充）
stride（步幅）
dilations（空洞值）
```
3. [多输入多输出的卷积过程](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)

```
当输入数据含多个通道时，我们需要构造一个与输入数据通道数相同的卷积核， 从而能够与含多通道的输入数据做互相关运算，得到一个输出通道的值；多个输出通道是多个卷积核与输入数据运算的结果。
```
- 池化
1. 作用：增加网络对旋转和位移的鲁棒性
2. 超参数

```
padding（填充）
stride（步幅）
dilation_rate(空洞值)

```
3. 分类

```
mean-pooling(平均值池化)，bp时将残差等比例传至上一层计算
max-pooling(最大值池化)，独立变量记录fp时的最大值位置，bp时残差传至上一层的最大值位置处，其他位置为0
```
- 激活函数