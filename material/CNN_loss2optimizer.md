## Optimizer
- 优化器的作用是通过损失函数降低训练误差，得到优化问题的数值解
- 与深度学习间的关系
```
优化算法降低训练误差；  
深度学习旨在降低泛化误差,为降低泛化误差，除了降低训练误差外，还应预防过拟合的发生
```
- 难点

1. 局部最小值的影响。当一个优化问题的数值解在局部最优解附近时， 由于目标函数有关解的梯度接近或变成零， 最终迭代求得的数值解可能只令目标函数局部最小化而非全局最小化。

```
区域内的最小值为局部最小值；
定义域内的最小值为全局最小值；
```
2. 鞍点的影响。深度学习模型参数通常都是高维的，目标函数的鞍点通常比局部最小值更常见。

```
鞍点直观理解为马鞍上可坐区域的中心。
```
3. 数学理论

```
当函数的海森矩阵在梯度为零的位置上的特征值全为正时，该函数得到局部最小值;
当函数的海森矩阵在梯度为零的位置上的特征值全为负时，该函数得到局部最大值;
当函数的海森矩阵在梯度为零的位置上的特征值有正有负时，该函数得到鞍点
```
- [梯度下降法](https://www.zybuluo.com/hanbingtao/note/448086)。梯度是一个向量，是函数变化率最大的方向，它指向函数值上升最快的方向。下降法的本质是泰勒级数，通过负梯度使目标函数的值减小。关于梯度与方向导数的关系可看[该文章](https://www.zhihu.com/question/36301367)
1. 使用适当的学习率，沿着梯度反方向更新自变量可能降低目标函数值
2. 学习率过大或过小都有问题。一个合适的学习率通常是需要通过多次实验找到的。

```
lr小时，会导致x更新缓慢从而需要更多的迭代才能得到较好的解；
lr大时，函数的一阶泰勒展开公式不再成立：这时我们无法保证迭代x会降低f(x)的值。
```
3. 当训练数据集的样本较多时，梯度下降每次迭代的计算开销较大，因而随机梯度下降通常更受青睐。
4. 分类：一维梯度下降法、多维梯度下降法和随机梯度下降法

```

一维函数的梯度是一个标量，也称导数;
多维梯度下降法依赖于方向导数(函数对任意方向的变化率).方向导数最大时函数值朝这个反向变化最快;
随机梯度每次迭代中只随机采样一个样本来计算梯度；随机梯度是梯度的无偏估计；基于随机采样得到的梯度的方差在迭代过程中无法减小，因此在实际中，（小批量）随机梯度下降的学习率可以在迭代过程中自我衰减，
```
- [动量法Momentum](http://flyrie.top/2018/09/20/Momentum/)

1. 出发点

```
自变量的移动方向与最优解的方向不一致
```
2. 两点改进。使得相邻时间步的自变量更新在方向上更加一致;使用了指数加权移动平均的思想。它将过去时间步的梯度做了加权平均，且权重按时间步指数衰减。

```
自变量在各个方向上的移动幅度不仅取决当前梯度， 还取决于过去的各个梯度在各个方向上是否一致;
相比于小批量随机梯度下降，动量法在每个时间步的自变量更新量近似于将前者对应的最近 1/(1−γ)个时间步的更新量做了指数加权移动平均后再除以1−γ 
```
3. 指数加权移动平均(exponentially weighted moving average）

```
当前时间步t的变量yt是由最近 1/(1−γ) (γ < 1)个时间步的另一个变量xt值的加权平均
离当前时间步t越近的x_t值获得的权重越大（越接近1）
```
- AdaGrad算法

1. 出发点

```
目标函数自变量的每一个元素在相同时间步都使用同一个学习率来自我迭代
```
2. 两点改进。在迭代过程中不断调整学习率，并让目标函数自变量中每个元素都分别拥有自己的学习率。使用AdaGrad算法时，自变量中每个元素的学习率在迭代过程中一直在降低（或不变）。

```
使用一个小批量随机梯度gt按元素平方的累加变量 st来对学习率进行修正；
目标函数自变量中每个元素的学习率通过st和初始学习率按元素运算重新调整一下；
```
- RMSProp算法

1. 出发点

```
当学习率在迭代早期降得较快且当前解依然不佳时，AdaGrad算法在迭代后期由于学习率过小，可能较难找到一个有用的解。
```
2. 改进方法。将梯度按元素平方做指数加权移动平均，这样，自变量每个元素的学习率在迭代过程中就不再一直降低（或不变）。
3. 与AdaGrad算法的不同之处
```
RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均来调整学习率。
```  
- AdaDelta算法

1. 出发点

```
AdaGrad算法在迭代后期可能较难找到有用解
```
2. 改进。没有学习率超参数，它通过使用有关自变量更新量平方的指数加权移动平均的项来替代RMSProp算法中的学习率。

```
维护一个额外的状态变量 Δxt，用来记录自变量变化量g′t按元素平方的指数加权移动平均
```
- Adam算法

1. 出发点

```
时间步t较小时，过去各时间步小批量随机梯度权值之和会较小
```
2. 改进项。

```
在RMSProp算法的基础上对小批量随机梯度也做了指数加权移动平均。
Adam算法使用了偏差修正。
```
## LOSS
- [fcn](https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py)
```
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)
```