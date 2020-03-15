## 基本概念

数学概念及符号

如果要定义一个函数 f(x)=x^{2}的话，Mathematica中的写法是这样的：

> In[28]:= f[x] := *x*^2



### 卷积

- [卷积](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
1. 作用：提取特征
2. 超参数:改变输出特征图的大小

> padding（填充）
> stride（步幅）
> dilations（空洞值）

3. [多输入多输出的卷积过程](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)

> 当输入数据含多个通道时，我们需要构造一个与输入数据通道数相同的卷积核， 从而能够与含多通道的输入数据做互相关运算，得到一个输出通道的值；多个输出通道是多个卷积核与输入数据运算的结果。

- 池化
1. 作用：增加网络对旋转和位移的鲁棒性
2. 超参数

> padding（填充）
> stride（步幅）
> dilation_rate(空洞值)

3. [分类及bp计算](http://www.voidcn.com/article/p-rbpamgzn-bee.html)

> mean-pooling(平均值池化)，bp时将残差等比例传至上一层计算
> max-pooling(最大值池化)，独立变量记录fp时的最大值位置，bp时残差传至上一层的最大值位置处，其他位置为0

- 激活函数
1. 作用：增强网络的表大能力
2. ReLU的优点

> 计算简单，没有sigmoid中的求幂运算；
> 模型更易训练，正值域的梯度恒为1；

3. sigmoid的缺点

> 函数值接近0或1，梯度几乎为0，造成bp时梯度消失，无法更新模型参数

- 感受野

> 影响元素x的前向计算的所有可能输入区域 （可能大于输入的实际尺寸）叫做 x 的感受野（receptive field）

- 特征图

> 二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征，也叫特征图（feature map）

- 连接，指不同层之间通过权重关联起来

> 当前层的连接数=该层特征图大小*通道数*（卷积核数量*卷积核大小^2+1）

- [参数量的计算](https://datascience.stackexchange.com/questions/17064/number-of-parameters-for-convolution-layers)=(输出通道数*（输入通道数*卷积核大小^2 + 1）)

> 局部连接；权值共享

### 网络模块及作用

#### 识别领域

- 1*1卷积

- [Batch Normalization](https://zh.d2l.ai/chapter_convolutional-neural-networks/batch-norm.html)
1. 出发点

> 深层神经网络中即使输入数据已做标准化，训练中模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。 这种计算数值的不稳定性通常令我们难以训练出有效的深度模型。

2. 作用

> 标准化处理输入数据使各个特征的分布相近，这往往更容易训练出有效的模型；
> BN层使深层神经网络的训练变得更加容易;
>  BN利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

3. 操作。训练阶段：batch设得大一点，从而使批量内样本的均值和方差的计算都较为准确;预测阶段：通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。

> 应用于激活函数之前；
> 对Conv层的每个通道分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数，并均为标量。

4. 由于归一化参数不同，导致BN层的训练和预测时的计算结果也是不一样的。预测时通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出

#### 检测领域

- high-res classifier
- high-res detector
- multi-scale
- passthrough，类似skip architecture
- anchor boxes
- residual blocks
- skip connections 
- upsampling

### 创新网络

#### [识别网络](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html)

- LeNet卷积神经网络

> 第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，增加输出通道使两个卷积层的参数尺寸类似

- AlexNet深度卷积神经网络

> 它首次证明了学习到的特征可以超越手工设计的特征

- VGG重复元素的网络

> 两个3\*3卷积核的感受野和1个5\*5卷积核的感受野相同，且参数减少；
> 通过使卷积核大小减半以及通道翻倍的设计，使得多数卷积层都有相同的模型参数尺寸和计算复杂度

- NIN网络中的网络

> 去除了容易造成过拟合的全连接输出层,将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层;
> 重复使用由卷积层和代替全连接层的1×1卷积层构成的NiN块来构建深层网络;

- Inception并行连接的网络

> 它通过不同窗口形状的卷积层和最大池化层来并行抽取信息;
> 使用1×1卷积层减少通道数从而降低模型复杂度;
> Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的;

- ResNet残差网络
1. 出发点

> 对神经网络添加新层后，理论上，原模型解的空间只是新模型解的空间的子空间。 因此添加层似乎更容易降低训练误差。实际上，即使加入BN处理， 添加过多的层后训练误差往往不降反升。

2. 名字由来：假设可通过堆叠的网络层近似逼近任意复杂的函数，从中学习输入x到输出的映射关系H，ResNet旨在学习另一个函数F(x)，使F(x):=H(x)– x，变换格式后得到H(x):=F(x)+x即残差模块的输出部分，由于x为恒等映射，所以网络实际学习的是一个残差函数F(x)。

> Instead of hoping each few stacked layers directly fit a desired underlying mapping,  we explicitly let these layers fit a residual mapping. Formally, denoting the desired underlying mapping as H(x), we let the stacked nonlinear
> layers fit another mapping of F(x) := H(x) - x. The original mapping is recast into F(x)+x. We hypothesize that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping. To the extreme, if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers.

> Let us consider H(x) as an underlying mapping to be fit by a few stacked layers (not necessarily the entire net), with x denoting the inputs to the first of these layers. If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated functions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, i.e., H(x) - x (assuming that the input and output are of the same dimensions). So rather than expect stacked layers to approximate H(x), we explicitly let these layers approximate a residual function F(x) := H(x) - x. The original function thus becomes F(x)+x [Deep Residual Learning for Image Recognition 2015]

3. 作用

> 在残差块中，输入可通过跨层的数据线路更快地向前传播。
> 残差映射在实际中往往更容易优化;

4. 操作

> 将浅层的数据与深层的数据相加

- DenseNet稠密连接网络
1. 出发点

> 靠近输入层与输出层之间的地方使用短连接（shorter_connections），就可以训练更深、更准确、更有效的卷积网络。

2. 组成模块：过渡层（transition layer)和稠密块（dense block）

> 稠密块定义了输入和输出是如何连结的。卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）;
> 过渡层用来控制通道数，使之不过大。通过 1×1 卷积层来减小通道数，使用步幅为2的平均池化层减半高和宽；

3. 与ResNet的差别是，ResNet跨层连接时使用相加的处理，DenseNet跨层连接时使用连结的处理逻辑
4. 优点

> 缓解梯度消失问题;
> 加强特征传播;
> 鼓励特征复用;
> 极大的减少了参数量;

5. 缺点:训练时十分消耗内存

> 运算需要，如对于大多数框架（如Torch和TensorFlow），每次拼接操作都会开辟新的内存来保存拼接后的特征；
> 一个 L 层的网络，要消耗相当 于L(L+1)/2层网络的内存（第 l 层的输出在内存里被存了 (L-l+1) 份

```

```
