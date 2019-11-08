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
3. [分类及bp计算](http://www.voidcn.com/article/p-rbpamgzn-bee.html)

```
mean-pooling(平均值池化)，bp时将残差等比例传至上一层计算
max-pooling(最大值池化)，独立变量记录fp时的最大值位置，bp时残差传至上一层的最大值位置处，其他位置为0
```
- 激活函数

1. 作用：增强网络的表大能力
2. ReLU的优点

```
计算简单，没有sigmoid中的求幂运算；
模型更易训练，正值域的梯度恒为1；
```
3. sigmoid的缺点

```
函数值接近0或1，梯度几乎为0，造成bp时梯度消失，无法更新模型参数
```
- 感受野

```
影响元素x的前向计算的所有可能输入区域 （可能大于输入的实际尺寸）叫做 x 的感受野（receptive field）
```
- 特征图

```
二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，也叫特征图（feature map）
```
- 连接，指不同层之间通过权重关联起来

```
当前层的连接数=该层特征图大小*通道数*（卷积核数量*卷积核大小^2+1）
```
- [参数量的计算](https://datascience.stackexchange.com/questions/17064/number-of-parameters-for-convolution-layers)=(输出通道数*（输入通道数*卷积核大小^2 + 1）)

```
局部连接；权值共享

```


