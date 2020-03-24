### 卷积

###### 参考资料

> [Understanding Memory Formats](https://intel.github.io/mkl-dnn/understanding_memory_formats.html)
> 
> [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
> 
> [cs231n\_convolution](http://cs231n.github.io/convolutional-networks/#conv)
> 
> [Intuitively Understanding Convolutions for Deep Learning](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)
> 
> [如何理解空洞卷积（dilated convolution）？](https://www.zhihu.com/question/54149221)
> 
> [What's the use of dilated convolutions?](https://stackoverflow.com/questions/41178576/whats-the-use-of-dilated-convolutions)
> 
> [卷积神经网络中十大拍案叫绝的操作](https://cloud.tencent.com/developer/article/1038802)
> 
> [ A guide to receptive field arithmetic for Convolutional Neural Networks](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

###### 基础理解



- 作用：提取特征

- 特点：局部连接、权值共享

- 超参数:改变输出特征图的大小(输出的空间布局)

> padding（填充）
> stride（步幅）
> dilations（空洞值）

- 卷积的过程

![img](../img/Basic_conv.gif)

- 多输入多输出的卷积过程，图片来自[这里](http://cs231n.github.io/convolutional-networks/#conv)。

![图片](../img/Basic_convs.gif)

> 当输入数据含多个通道时，我们需要构造一个与输入数据通道数相同的卷积核， 从而能够与含多通道的输入数据做互相关运算，得到一个输出通道的值；多个输出通道是多个卷积核与输入数据运算的结果。

- 感受野

> 影响元素x的前向计算的所有可能输入区域 （可能大于输入的实际尺寸）叫做 x 的感受野（receptive field）
> 
> 卷积神经网络每一层输出的特征图（feature map）上的像素点在原始图像上映射的区域大小

- 输出特征图大小与感受野计算

> K：卷积核大小
> 
> P：填充大小
> 
> S：移动步长
> 
> N：特征图大小
> 
> R：感受野大小
> 
> J：相邻像素间的步长
> 
> - 输出特征图大小
> 
> $$
> N_{out} = \left \lfloor (N_{in} - K + 2P)/S \right \rfloor + 1
> $$
> 
> - 感受野大小 
> 
> $$
> R_{0} = K \newline
> J_{0} = S \newline
> R_{out}=R_{in} + (K-1)*J_{in} \newline
> J_{out}=J_{in}*S
> 
> $$
> 
> - 空洞卷积的大小
> 
> pass

- 特征图、感受野、第一个输出特征的感受野的中心位置，图片来自[这里](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

![img](../img/Basic_FeatureMap_RF.png)

###### 

- [deconvolution](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_ops.py) deconv只是观念上和传统的conv反向(但是不是更新梯度的反向传播)),传统的conv是从图片生成feature map，而deconv是用unsupervised的方法找到一组kernel和feature map，让它们重建图片;

- [deconv和conv关系](the actual weight values in the matrix does not have to come from the original convolution matrix. What’s important is that the weight layout is transposed from that of the convolution matrix)conv是多对1，deconv是1对多；位置关联；转置卷积的权重是学出来的;转置卷积可以得到原来的图像大小，但实际使用中转置卷积核的大小无关紧要，因为转置卷积参数中有输出大小的参数
  
  > the actual weight values in the matrix does not have to come from the original convolution matrix. What’s important is that the weight layout is transposed from that of the convolution matrix

- [deconv相关介绍](https://www.zhihu.com/question/43609045)

###### 池化

![img](../img/Basic_MaxPooling.jpg)

                                                                        图片来自[这里](http://cs231n.github.io/convolutional-networks/#conv)

- 作用：增加网络对旋转和位移的鲁棒性

- 超参数

> padding（填充）
> stride（步幅）
> dilation_rate(空洞值)

- [分类及bp计算](http://www.voidcn.com/article/p-rbpamgzn-bee.html)

> average-pooling(平均值池化)，bp时将残差等比例传至上一层计算
> max-pooling(最大值池化)，独立变量记录fp时的最大值位置，bp时残差传至上一层的最大值位置处，其他位置为0
> 
> global-pooling (全局池化), 包括全局平均池化和全局最大化池化，旨在将整个特征图映射为1个值。

###### 卷积可视化

- 特点：以热量图的形式表示数值大小或正确分类的高低

- 分类：参数(滤波核)可视化和非参数(特征图)可视化

- 作用

> 观察特征图的响应。如ZFnet根据特征可视化，提出AlexNet第一个卷积层卷积核太大，导致提取到的特征模糊；
> 通过每层特征图的变化得到特征演变过程；
> 对分类器进行敏感性分析。可通过阻止部分输入图像揭示那部分对分类是重要的；
> 诊断模型潜在问题；

- [卷积核及特征图可视化参考代码](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/)

##### 分类

**卷积核的4个超参数：输入特征图的通道数、滤波器的高度、滤波器的宽度和输出特征图的通道数**。不同类型的卷积及其计算量均是通过上述4个超参数间的组合变换得到的，其中计算量可看下图，图片来自[这里](https://medium.com/@yu4u/why-mobilenet-and-its-variants-e-g-shufflenet-are-fast-1c7048b9618d)。![img](../img/Model_conv_calucation.png)

###### 基础卷积

- 数据格式：NCHW和NHWC，在内存中的存储方式也不同。图片来自[这里](https://intel.github.io/mkl-dnn/understanding_memory_formats.html)。

> **N**: number of images in the batch
> 
> **H**: height of the image
> 
> **W**: width of the image
> 
> **C**: number of channels of the image (ex: 3 for RGB, 1 for grayscale...)
> 
> TensorFlow 为什么选择 NHWC 格式作为默认格式？因为早期开发都是基于 CPU，使用 NHWC 比 NCHW 稍快一些（不难理解，NHWC 局部性更好，cache 利用率高）。
> 
> NCHW 则是 Nvidia cuDNN 默认格式，使用 GPU 加速时用 NCHW 格式速度会更快（也有个别情况例外）。
> 
> 最佳实践：设计网络时充分考虑两种格式，最好能灵活切换，在 GPU 上训练时使用 NCHW 格式，在 CPU 上做预测时使用 NHWC 格式。

![img](../img/Basic_Data_layout.png)

- 标准卷积

> 卷积核格式类似于特征图数据格式，单个卷积核的大小为HWC，其中H和W为卷积核大小，C为输入特征图的通道数。卷积操作时将每个通道上卷积的结果相加作为输出特征图上对应位置的值。对于输出M个特征图时，需要M个HWC维度的卷积核分别卷积。

- 反卷积

- 膨胀卷积

- 可变形卷积

- 点卷积

- 深度卷积

- 通道shuffle卷积

###### 组合卷积

- 分组卷积

- bottleneck

- 深度可分离卷积

- inception

###### 卷积+运算

- 残差模块

- 反残差模块

- 稠密卷积

###### 参考代码

- [可分离卷积](https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/contrib/layers/python/layers/layers.py)

```
def separable_convolution2d(
    inputs,
    num_outputs,
    kernel_size,
    depth_multiplier=1,
    stride=1,
    padding='SAME',
    data_format=DATA_FORMAT_NHWC,
    rate=1,
    activation_fn=nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    pointwise_initializer=None,
    weights_regularizer=None,
    biases_initializer=init_ops.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None):
  """Adds a depth-separable 2D convolution with optional batch_norm layer.
  This op first performs a depthwise convolution that acts separately on
  channels, creating a variable called `depthwise_weights`. If `num_outputs`
  is not None, it adds a pointwise convolution that mixes channels, creating a
  variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
  it adds bias to the result, creating a variable called 'biases', otherwise,
  the `normalizer_fn` is applied. It finally applies an activation function
  to produce the end result.
  Args:
    inputs: A tensor of size [batch_size, height, width, channels].
    num_outputs: The number of pointwise convolution output filters. If is None,
      then we skip the pointwise convolution stage.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of of the
      filters. Can be an int if both values are the same.
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `num_filters_in * depth_multiplier`.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      depthwise convolution stride. Can be an int if both strides are the same.
    padding: One of 'VALID' or 'SAME'.
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    rate: A list of length 2: [rate_height, rate_width], specifying the dilation
      rates for atrous convolution. Can be an int if both rates are the same. If
      any value is larger than one, then both stride values need to be one.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the depthwise weights.
    pointwise_initializer: An initializer for the pointwise weights. default set
      to None, means use weights_initializer.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: Whether or not the variables should be trainable or not.
    scope: Optional scope for variable_scope.
  Returns:
    A `Tensor` representing the output of the operation.
  Raises:
    ValueError: If `data_format` is invalid.
  """
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')
  layer_variable_getter = _build_variable_getter({
      'bias': 'biases',
      'depthwise_kernel': 'depthwise_weights',
      'pointwise_kernel': 'pointwise_weights'
  })

  with variable_scope.variable_scope(
      scope,
      'SeparableConv2d', [inputs],
      reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)

    if pointwise_initializer is None:
      pointwise_initializer = weights_initializer

    df = ('channels_first'
          if data_format and data_format.startswith('NC') else 'channels_last')
    if num_outputs is not None:
      # Apply separable conv using the SeparableConvolution2D layer.
      layer = convolutional_layers.SeparableConvolution2D(
          filters=num_outputs,
          kernel_size=kernel_size,
          strides=stride,
          padding=padding,
          data_format=df,
          dilation_rate=utils.two_element_tuple(rate),
          activation=None,
          depth_multiplier=depth_multiplier,
          use_bias=not normalizer_fn and biases_initializer,
          depthwise_initializer=weights_initializer,
          pointwise_initializer=pointwise_initializer,
          bias_initializer=biases_initializer,
          depthwise_regularizer=weights_regularizer,
          pointwise_regularizer=weights_regularizer,
          bias_regularizer=biases_regularizer,
          activity_regularizer=None,
          trainable=trainable,
          name=sc.name,
          dtype=inputs.dtype.base_dtype,
          _scope=sc,
          _reuse=reuse)
      outputs = layer.apply(inputs)

      # Add variables to collections.
      _add_variable_to_collections(layer.depthwise_kernel,
                                   variables_collections, 'weights')
      _add_variable_to_collections(layer.pointwise_kernel,
                                   variables_collections, 'weights')
      if layer.bias is not None:
        _add_variable_to_collections(layer.bias, variables_collections,
                                     'biases')

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
    else:
      # Actually apply depthwise conv instead of separable conv.
      dtype = inputs.dtype.base_dtype
      kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
      stride_h, stride_w = utils.two_element_tuple(stride)
      num_filters_in = utils.channel_dimension(
          inputs.get_shape(), df, min_rank=4)
      weights_collections = utils.get_variable_collections(
          variables_collections, 'weights')

      depthwise_shape = [kernel_h, kernel_w, num_filters_in, depth_multiplier]
      depthwise_weights = variables.model_variable(
          'depthwise_weights',
          shape=depthwise_shape,
          dtype=dtype,
          initializer=weights_initializer,
          regularizer=weights_regularizer,
          trainable=trainable,
          collections=weights_collections)
      strides = [
          1, 1, stride_h, stride_w
      ] if data_format.startswith('NC') else [1, stride_h, stride_w, 1]

      outputs = nn.depthwise_conv2d(
          inputs,
          depthwise_weights,
          strides,
          padding,
          rate=utils.two_element_tuple(rate),
          data_format=data_format)
      num_outputs = depth_multiplier * num_filters_in

      if normalizer_fn is not None:
        normalizer_params = normalizer_params or {}
        outputs = normalizer_fn(outputs, **normalizer_params)
      else:
        if biases_initializer is not None:
          biases_collections = utils.get_variable_collections(
              variables_collections, 'biases')
          biases = variables.model_variable(
              'biases',
              shape=[
                  num_outputs,
              ],
              dtype=dtype,
              initializer=biases_initializer,
              regularizer=biases_regularizer,
              trainable=trainable,
              collections=biases_collections)
          outputs = nn.bias_add(outputs, biases, data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
```

- 深度可分离卷积([depthwise_separable_conv](https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py))

```
  def _depthwise_separable_conv(inputs,
                            num_pwc_filters,
                            width_multiplier,
                            sc,
                            downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv')

    bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv')
    bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn
```
