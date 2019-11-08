## 基本概念
### 卷积
- [卷积](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
1. 作用：提取特征
2. 超参数:改变输出特征图的大小

```
padding（填充）
stride（步幅）
```
3. [多输入多输出的卷积过程](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)

```
当输入数据含多个通道时，我们需要构造一个与输入数据通道数相同的卷积核， 从而能够与含多通道的输入数据做互相关运算，得到一个输出通道的值；多个输出通道是多个卷积核与输入数据运算的结果。
```
- 池化