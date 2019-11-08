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