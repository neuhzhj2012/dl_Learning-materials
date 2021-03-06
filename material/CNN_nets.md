## 文章

- [传统检测和深度学习检测流程](https://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-object-detection-upc-2016)

- [Review: DenseNet — Dense Convolutional Network (Image Classification)](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

- [Understanding Feature Pyramid Networks for object detection (FPN)](https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

- [mobilenetv2-inverted-residuals-and-linear-bottlenecks](https://www.slideshare.net/JinwonLee9/pr108-mobilenetv2-inverted-residuals-and-linear-bottlenecks)

## 网络结构

- [lenet](https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb) tf基础函数版

```
def LeNet(x):    
  # Hyperparameters
  mu = 0
  sigma = 0.1
  layer_depth = {
      'layer_1' : 6,
      'layer_2' : 16,
      'layer_3' : 120,
      'layer_f1' : 84
  }
```

# TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.

  conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
  conv1_b = tf.Variable(tf.zeros(6))
  conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 

# TODO: Activation.

  conv1 = tf.nn.relu(conv1)

# TODO: Pooling. Input = 28x28x6. Output = 14x14x6.

  pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

# TODO: Layer 2: Convolutional. Output = 10x10x16.

  conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
  conv2_b = tf.Variable(tf.zeros(16))
  conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b

# TODO: Activation.

  conv2 = tf.nn.relu(conv2)

# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.

  pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 

# TODO: Flatten. Input = 5x5x16. Output = 400.

  fc1 = flatten(pool_2)

# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.

  fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
  fc1_b = tf.Variable(tf.zeros(120))
  fc1 = tf.matmul(fc1,fc1_w) + fc1_b

# TODO: Activation.

  fc1 = tf.nn.relu(fc1)

# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.

  fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
  fc2_b = tf.Variable(tf.zeros(84))
  fc2 = tf.matmul(fc1,fc2_w) + fc2_b

# TODO: Activation.

  fc2 = tf.nn.relu(fc2)

# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.

  fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
  fc3_b = tf.Variable(tf.zeros(10))
  logits = tf.matmul(fc2, fc3_w) + fc3_b
  return logits

```
- [lenet](https://github.com/Zehaos/MobileNet/blob/master/nets/lenet.py) slim集成函数版
```

def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
  end_points = {}

  with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    net = slim.flatten(net)
    end_points['Flatten'] = net

    net = slim.fully_connected(net, 1024, scope='fc3')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                  scope='fc4')

  end_points['Logits'] = logits
  end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
lenet.default_image_size = 28

```
- [cifarnet](https://github.com/Zehaos/MobileNet/blob/master/nets/cifarnet.py)带LRN和dropout的FC网络
```

def cifarnet(images, num_classes=10,is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet'):
  end_points = {}

  with tf.variable_scope(scope, 'CifarNet', [images, num_classes]):
    net = slim.conv2d(images, 64, [5, 5], scope='conv1')
    end_points['conv1'] = net
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
    end_points['pool1'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    end_points['conv2'] = net
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
    end_points['pool2'] = net
    net = slim.flatten(net)
    end_points['Flatten'] = net
    net = slim.fully_connected(net, 384, scope='fc3')
    end_points['fc3'] = net
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout3')
    net = slim.fully_connected(net, 192, scope='fc4')
    end_points['fc4'] = net
    logits = slim.fully_connected(net, num_classes,
                                  biases_initializer=tf.zeros_initializer(),
                                  weights_initializer=trunc_normal(1/192.0),
                                  weights_regularizer=None,
                                  activation_fn=None,
                                  scope='logits')

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
cifarnet.default_image_size = 32

```

- [Alexnet](https://github.com/Zehaos/MobileNet/blob/master/nets/alexnet.py)全卷积结构
```

with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                    outputs_collections=[end_points_collection]):
  net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                    scope='conv1')
  net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
  net = slim.conv2d(net, 192, [5, 5], scope='conv2')
  net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')#注意pool2操作的卷积核大小为3，默认卷积padding='valid',所以得到的特征图为(26-3)/2+1=12
  net = slim.conv2d(net, 384, [3, 3], scope='conv3')
  net = slim.conv2d(net, 384, [3, 3], scope='conv4')
  net = slim.conv2d(net, 256, [3, 3], scope='conv5')
  net = slim.max_pool2d(net, [3, 3], 2, scope='pool5') 

# Use conv2d instead of fully_connected layers.

  with slim.arg_scope([slim.conv2d],
                      weights_initializer=trunc_normal(0.005),
                      biases_initializer=tf.constant_initializer(0.1)):
    net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                      scope='fc6')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout6')
    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                       scope='dropout7')
    net = slim.conv2d(net, num_classes, [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      biases_initializer=tf.zeros_initializer(),
                      scope='fc8')

# Convert end_points_collection into a end_point dict.

  end_points = slim.utils.convert_collection_to_dict(end_points_collection)
  if spatial_squeeze:
    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    end_points[sc.name + '/fc8'] = net
  return net, end_points
alexnet_v2.default_image_size = 224            

```
- 网络各特征图大小
```

alexnet_v2
alexnet_v2/conv1/Relu shape(5, 54, 54, 64)
alexnet_v2/pool1/MaxPool shape(5, 26, 26, 64)
alexnet_v2/conv2/Relu shape(5, 26, 26, 192)
alexnet_v2/pool2/MaxPool shape(5, 12, 12, 192)
alexnet_v2/conv3/Relu shape(5, 12, 12, 384)
alexnet_v2/conv4/Relu shape(5, 12, 12, 384)
alexnet_v2/conv5/Relu shape(5, 12, 12, 256)
alexnet_v2/pool5/MaxPool shape(5, 5, 5, 256)
alexnet_v2/fc6/Relu shape(5, 1, 1, 4096)
alexnet_v2/dropout6/dropout/mul_1 shape(5, 1, 1, 4096)
alexnet_v2/fc7/Relu shape(5, 1, 1, 4096)
alexnet_v2/dropout7/dropout/mul_1 shape(5, 1, 1, 4096)
alexnet_v2/fc8/BiasAdd shape(5, 1, 1, 1000)

```
- VGG16网络结构。重复使用相同卷积；5*5的卷积使用2个3*3的卷积替代
```

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_16.default_image_size = 224

```
- vgg16网络各特征图的大小
```

vgg_16/conv1/conv1_1 (5, 224, 224, 64)
vgg_16/conv1/conv1_2 (5, 224, 224, 64)
vgg_16/pool1 (5, 112, 112, 64)
vgg_16/conv2/conv2_1 (5, 112, 112, 128)
vgg_16/conv2/conv2_2 (5, 112, 112, 128)
vgg_16/pool2 (5, 56, 56, 128)
vgg_16/conv3/conv3_1 (5, 56, 56, 256)
vgg_16/conv3/conv3_2 (5, 56, 56, 256)
vgg_16/conv3/conv3_3 (5, 56, 56, 256)
vgg_16/pool3 (5, 28, 28, 256)
vgg_16/conv4/conv4_1 (5, 28, 28, 512)
vgg_16/conv4/conv4_2 (5, 28, 28, 512)
vgg_16/conv4/conv4_3 (5, 28, 28, 512)
vgg_16/pool4 (5, 14, 14, 512)
vgg_16/conv5/conv5_1 (5, 14, 14, 512)
vgg_16/conv5/conv5_2 (5, 14, 14, 512)
vgg_16/conv5/conv5_3 (5, 14, 14, 512)
vgg_16/pool5 (5, 7, 7, 512)
vgg_16/fc6 (5, 1, 1, 4096)
vgg_16/fc7 (5, 1, 1, 4096)
vgg_16/fc8 (5, 1000)

```
- inception结构，展示代码为[v1](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v1.py)结构的Mixed_3b层。一方面增加了网络的width，另一方面增加了网络对尺度的适应性；
```

end_point = 'Mixed_3b'
with tf.variable_scope(end_point):
  with tf.variable_scope('Branch_0'):
    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
  with tf.variable_scope('Branch_1'):
    branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
    branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
  with tf.variable_scope('Branch_2'):
    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
    branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
  with tf.variable_scope('Branch_3'):
    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
  net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
end_points[end_point] = net

```
- v1的完整结构
```

def inception_v1_base(inputs,
                      final_endpoint='Mixed_5c',
                      scope='InceptionV1'):
  """Defines the Inception V1 base architecture.

    Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01)):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)    #该scope中的padding设置为SAME,故输出特征图仅和stride有关
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    
        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV1'):
  """Defines the Inception V1 architecture.

  This architecture is defined in:
  http://arxiv.org/pdf/1409.4842v1.pdf.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

# Final pooling and prediction

  with tf.variable_scope(scope, 'InceptionV1', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [7, 7], stride=1, scope='AvgPool_0a_7x7')
        net = slim.dropout(net,
                           dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
inception_v1.default_image_size = 224

```
- inception_V1网络各特征图大小
```

InceptionV1
InceptionV1/InceptionV1/Conv2d_1a_7x7/Relu shape(5, 112, 112, 64)   
InceptionV1/InceptionV1/MaxPool_2a_3x3/MaxPool shape(5, 56, 56, 64)   
InceptionV1/InceptionV1/Conv2d_2b_1x1/Relu shape(5, 56, 56, 64)
InceptionV1/InceptionV1/Conv2d_2c_3x3/Relu shape(5, 56, 56, 192)
InceptionV1/InceptionV1/MaxPool_3a_3x3/MaxPool shape(5, 28, 28, 192)
InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu shape(5, 28, 28, 64)
InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu shape(5, 28, 28, 96)
InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/Relu shape(5, 28, 28, 128)
InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu shape(5, 28, 28, 16)
InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/Relu shape(5, 28, 28, 32)
InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool shape(5, 28, 28, 192)
InceptionV1/InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/Relu shape(5, 28, 28, 32)
InceptionV1/InceptionV1/Mixed_3b/concat shape(5, 28, 28, 256)
InceptionV1/InceptionV1/Mixed_3c/concat shape(5, 28, 28, 480)
InceptionV1/InceptionV1/MaxPool_4a_3x3/MaxPool shape(5, 14, 14, 480)
InceptionV1/InceptionV1/Mixed_4b/concat shape(5, 14, 14, 512)
InceptionV1/InceptionV1/Mixed_4c/concat shape(5, 14, 14, 512)
InceptionV1/InceptionV1/Mixed_4d/concat shape(5, 14, 14, 512)
InceptionV1/InceptionV1/Mixed_4e/concat shape(5, 14, 14, 528)
InceptionV1/InceptionV1/Mixed_4f/concat shape(5, 14, 14, 832)
InceptionV1/InceptionV1/MaxPool_5a_2x2/MaxPool shape(5, 7, 7, 832)
InceptionV1/InceptionV1/Mixed_5b/concat shape(5, 7, 7, 832)
InceptionV1/InceptionV1/Mixed_5c/concat shape(5, 7, 7, 1024)
InceptionV1/InceptionV1/Mixed_5c/concat shape(5, 7, 7, 1024)
InceptionV1/Logits/AvgPool_0a_7x7/AvgPool shape(5, 1, 1, 1024)
InceptionV1/Logits/Dropout_0b/dropout/mul_1 shape(5, 1, 1, 1024)
InceptionV1/Logits/Conv2d_0c_1x1/BiasAdd shape(5, 1, 1, 1000)
InceptionV1/Logits/SpatialSqueeze shape(5, 1000)

```
- inception结构的[v2](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v2.py)版本中Mixed_3b层。使用两个3*3卷积替代5*5卷积，减少参数量。
```

end_point = 'Mixed_3b'
with tf.variable_scope(end_point):
with tf.variable_scope('Branch_0'):
  branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
with tf.variable_scope('Branch_1'):
  branch_1 = slim.conv2d(
      net, depth(64), [1, 1],
      weights_initializer=trunc_normal(0.09),
      scope='Conv2d_0a_1x1')
  branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                         scope='Conv2d_0b_3x3')
with tf.variable_scope('Branch_2'):
  branch_2 = slim.conv2d(
      net, depth(64), [1, 1],
      weights_initializer=trunc_normal(0.09),
      scope='Conv2d_0a_1x1')
  branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                         scope='Conv2d_0b_3x3')
  branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                         scope='Conv2d_0c_3x3')
with tf.variable_scope('Branch_3'):
  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
  branch_3 = slim.conv2d(
      branch_3, depth(32), [1, 1],
      weights_initializer=trunc_normal(0.1),
      scope='Conv2d_0b_1x1')
net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
end_points[end_point] = net

```

- inception_V2的完整结构，可分离卷积。
```

def inception_v2_base(inputs,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception v2 (6a2).    
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

# end_points will collect relevant activations for external use, for example

# summaries or losses.

  end_points = {}

# Used to find thinned depths for each layer.

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV2', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d, slim.separable_conv2d],
        stride=1, padding='SAME'):

      # Note that sizes in the comments below assume an input spatial size of
      # 224x224, however, the inputs can be of any size greater 32x32.
    
      # 224 x 224 x 3
      end_point = 'Conv2d_1a_7x7'
      # depthwise_multiplier here is different from depth_multiplier.
      # depthwise_multiplier determines the output channels of the initial
      # depthwise conv (see docs for tf.nn.separable_conv2d), while
      # depth_multiplier controls the # channels of the subsequent 1x1
      # convolution. Must have
      #   in_channels * depthwise_multipler <= out_channels
      # so that the separable convolution is not overparameterized.
      depthwise_multiplier = min(int(depth(64) / 3), 8)
      net = slim.separable_conv2d(
          inputs, depth(64), [7, 7], depth_multiplier=depthwise_multiplier,
          stride=2, weights_initializer=trunc_normal(1.0),
          scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 112 x 112 x 64
      end_point = 'MaxPool_2a_3x3'
      net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 56 x 56 x 64
      end_point = 'Conv2d_2b_1x1'
      net = slim.conv2d(net, depth(64), [1, 1], scope=end_point,
                        weights_initializer=trunc_normal(0.1))
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 56 x 56 x 64
      end_point = 'Conv2d_2c_3x3'
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 56 x 56 x 192
      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v2(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV2'):
  """Inception v2 model for classification.

  Constructs an Inception v2 network for classification as described in
  http://arxiv.org/abs/1502.03167.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

# Final pooling and prediction

  with tf.variable_scope(scope, 'InceptionV2', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v2_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)
      with tf.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v2.default_image_size = 224

```

- inception_V2网络各特征图大小
```

InceptionV2
InceptionV2/InceptionV2/Conv2d_1a_7x7/Relu (5, 112, 112, 64)
InceptionV2/InceptionV2/MaxPool_2a_3x3/MaxPool (5, 56, 56, 64)
InceptionV2/InceptionV2/Conv2d_2b_1x1/Relu (5, 56, 56, 64)
InceptionV2/InceptionV2/Conv2d_2c_3x3/Relu (5, 56, 56, 192)
InceptionV2/InceptionV2/MaxPool_3a_3x3/MaxPool (5, 28, 28, 192)
InceptionV2/InceptionV2/Mixed_3b/concat (5, 28, 28, 256)
InceptionV2/InceptionV2/Mixed_3c/concat (5, 28, 28, 320)
InceptionV2/InceptionV2/Mixed_4a/concat (5, 14, 14, 576)
InceptionV2/InceptionV2/Mixed_4b/concat (5, 14, 14, 576)
InceptionV2/InceptionV2/Mixed_4c/concat (5, 14, 14, 576)
InceptionV2/InceptionV2/Mixed_4d/concat (5, 14, 14, 576)
InceptionV2/InceptionV2/Mixed_4e/concat (5, 14, 14, 576)
InceptionV2/InceptionV2/Mixed_5a/concat (5, 7, 7, 1024)
InceptionV2/InceptionV2/Mixed_5b/concat (5, 7, 7, 1024)
InceptionV2/InceptionV2/Mixed_5c/concat (5, 7, 7, 1024)
InceptionV2/Logits/AvgPool_1a_7x7/AvgPool (5, 1, 1, 1024)
InceptionV2/Logits/Dropout_1b/dropout/mul_1 (5, 1, 1, 1024)
InceptionV2/Logits/Conv2d_1c_1x1/BiasAdd (5, 1, 1, 1000)
InceptionV2/Logits/SpatialSqueeze (5, 1000)

```
- inception结构[V3](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v3.py)版本中Mixed_6b层。将7*7卷积核拆分为1*7和7*1的两层卷积达到相同感受野，较少参数量，增加了网络深度，提升了非线性表达;输入图像分辨率有224*224变为299*299;同时训练两个不同深度的网络，辅助分类器，提升浅层网络的表达能力。
```

end_point = 'Mixed_6b'
with tf.variable_scope(end_point):
with tf.variable_scope('Branch_0'):
  branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
with tf.variable_scope('Branch_1'):
  branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
  branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                         scope='Conv2d_0b_1x7')
  branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                         scope='Conv2d_0c_7x1')
with tf.variable_scope('Branch_2'):
  branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
  branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                         scope='Conv2d_0b_7x1')
  branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                         scope='Conv2d_0c_1x7')
  branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                         scope='Conv2d_0d_7x1')
  branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                         scope='Conv2d_0e_1x7')
with tf.variable_scope('Branch_3'):
  branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
  branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                         scope='Conv2d_0b_1x1')
net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
end_points[end_point] = net

```
- inception_v3的完整结构
```

def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.
  Here is a mapping from the old_names to the new names:
  Old name          | New name
=======================================

  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

# end_points will collect relevant activations for external use, for example

# summaries or losses.

  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):
      # 299 x 299 x 3
      end_point = 'Conv2d_1a_3x3'
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 149 x 149 x 32
      end_point = 'Conv2d_2a_3x3'
      net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 32
      end_point = 'Conv2d_2b_3x3'
      net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 64
      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 64
      end_point = 'Conv2d_3b_1x1'
      net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 80.
      end_point = 'Conv2d_4a_3x3'
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 71 x 71 x 192.
      end_point = 'MaxPool_5a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 35 x 35 x 192.

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # mixed: 35 x 35 x 256.
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_1: 35 x 35 x 288.
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                 scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_2: 35 x 35 x 288.
      end_point = 'Mixed_5d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_3: 17 x 17 x 768.
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed4: 17 x 17 x 768.
      end_point = 'Mixed_6b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_5: 17 x 17 x 768.
      end_point = 'Mixed_6c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # mixed_6: 17 x 17 x 768.
      end_point = 'Mixed_6d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_7: 17 x 17 x 768.
      end_point = 'Mixed_6e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_8: 8 x 8 x 1280.
      end_point = 'Mixed_7a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # mixed_9: 8 x 8 x 2048.
      end_point = 'Mixed_7b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    
      # mixed_10: 8 x 8 x 2048.
      end_point = 'Mixed_7c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
  """Inception model from http://arxiv.org/abs/1512.00567.

  "Rethinking the Inception Architecture for Computer Vision"

  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.

  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.

  The default image size used to train this network is 299x299.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v3_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      # Auxiliary Head logits
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        aux_logits = end_points['Mixed_6e']
        with tf.variable_scope('AuxLogits'):
          aux_logits = slim.avg_pool2d(
              aux_logits, [5, 5], stride=3, padding='VALID',
              scope='AvgPool_1a_5x5')
          aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                   scope='Conv2d_1b_1x1')
    
          # Shape of feature map before the final layer.
          kernel_size = _reduced_kernel_size_for_small_input(
              aux_logits, [5, 5])
          aux_logits = slim.conv2d(
              aux_logits, depth(768), kernel_size,
              weights_initializer=trunc_normal(0.01),
              padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
          aux_logits = slim.conv2d(
              aux_logits, num_classes, [1, 1], activation_fn=None,
              normalizer_fn=None, weights_initializer=trunc_normal(0.001),
              scope='Conv2d_2b_1x1')
          if spatial_squeeze:
            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
          end_points['AuxLogits'] = aux_logits
    
      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
        net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                              scope='AvgPool_1a_{}x{}'.format(*kernel_size))
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  return logits, end_points
inception_v3.default_image_size = 299

```
- inception_V3网络Aux各特征图的大小
```

InceptionV3
InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool (5, 73, 73, 64)
InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu (5, 73, 73, 80)
InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu (5, 71, 71, 192)
InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool (5, 35, 35, 192)
InceptionV3/InceptionV3/Mixed_5b/concat (5, 35, 35, 256)
InceptionV3/InceptionV3/Mixed_5c/concat (5, 35, 35, 288)
InceptionV3/InceptionV3/Mixed_5d/concat (5, 35, 35, 288)
InceptionV3/InceptionV3/Mixed_6a/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6b/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6c/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6d/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6e/concat (5, 17, 17, 768)
InceptionV3/AuxLogits/AvgPool_1a_5x5/AvgPool (5, 5, 5, 768)
InceptionV3/AuxLogits/Conv2d_1b_1x1/Relu (5, 5, 5, 128)
InceptionV3/AuxLogits/Conv2d_2a_5x5/Relu (5, 1, 1, 768)
InceptionV3/AuxLogits/Conv2d_2b_1x1/BiasAdd (5, 1, 1, 1000)
InceptionV3/AuxLogits/SpatialSqueeze (5, 1000)

```
- inception_V3主网络各特征图的大小
```

InceptionV3
InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool (5, 73, 73, 64)
InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu (5, 73, 73, 80)
InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu (5, 71, 71, 192)
InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool (5, 35, 35, 192)
InceptionV3/InceptionV3/Mixed_5b/concat (5, 35, 35, 256)
InceptionV3/InceptionV3/Mixed_5c/concat (5, 35, 35, 288)
InceptionV3/InceptionV3/Mixed_5d/concat (5, 35, 35, 288)
InceptionV3/InceptionV3/Mixed_6a/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6b/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6c/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6d/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_6e/concat (5, 17, 17, 768)
InceptionV3/InceptionV3/Mixed_7a/concat (5, 8, 8, 1280)
InceptionV3/InceptionV3/Mixed_7b/concat (5, 8, 8, 2048)
InceptionV3/InceptionV3/Mixed_7c/concat (5, 8, 8, 2048)
InceptionV3/Logits/AvgPool_1a_8x8/AvgPool (5, 1, 1, 2048)
InceptionV3/Logits/Dropout_1b/dropout/mul_1 (5, 1, 1, 2048)
InceptionV3/Logits/Conv2d_1c_1x1/BiasAdd (5, 1, 1, 1000)
InceptionV3/Logits/SpatialSqueeze (5, 1000)

```
- inception结构的[v4](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v4.py)版本中block_inception_c模块，branch2分支内部组合数据。发现ResNet结构可以加速模型的训练。
```

def block_inception_c(inputs, scope=None, reuse=None):
  """Builds Inception-C block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = tf.concat(axis=3, values=[
            slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
            slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
        branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
        branch_2 = tf.concat(axis=3, values=[
            slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
            slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

```
- inception_V4的完整结构
```

def block_inception_a(inputs, scope=None, reuse=None):
  """Builds Inception-A block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def block_reduction_a(inputs, scope=None, reuse=None):
  """Builds Reduction-A block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                               scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
        branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

def block_inception_b(inputs, scope=None, reuse=None):
  """Builds Inception-B block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
        branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
        branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
        branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def block_reduction_b(inputs, scope=None, reuse=None):
  """Builds Reduction-B block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
        branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
        branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
        branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                               padding='VALID', scope='Conv2d_1a_3x3')
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                   scope='MaxPool_1a_3x3')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

def block_inception_c(inputs, scope=None, reuse=None):
  """Builds Inception-C block for Inception v4 network."""

# By default use stride=1 and SAME padding

  with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                      stride=1, padding='SAME'):
    with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
      with tf.variable_scope('Branch_0'):
        branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
      with tf.variable_scope('Branch_1'):
        branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_1 = tf.concat(axis=3, values=[
            slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
            slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
      with tf.variable_scope('Branch_2'):
        branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
        branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
        branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
        branch_2 = tf.concat(axis=3, values=[
            slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
            slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
      with tf.variable_scope('Branch_3'):
        branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
        branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
      return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

def inception_v4_base(inputs, final_endpoint='Mixed_7d', scope=None):
  """Creates the Inception V4 network up to the given final endpoint.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    final_endpoint: specifies the endpoint to construct the network up to.
      It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
      'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
      'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
      'Mixed_7d']
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
  """
  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionV4', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 299 x 299 x 3
      net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                        padding='VALID', scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
      # 149 x 149 x 32
      net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 32
      net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 147 x 147 x 64
      with tf.variable_scope('Mixed_3a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_0a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_0a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_3a', net): return net, end_points

      # 73 x 73 x 160
      with tf.variable_scope('Mixed_4a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                 scope='Conv2d_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_4a', net): return net, end_points
    
      # 71 x 71 x 192
      with tf.variable_scope('Mixed_5a'):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1])
        if add_and_check_final('Mixed_5a', net): return net, end_points
    
      # 35 x 35 x 384
      # 4 x Inception-A blocks
      for idx in range(4):
        block_scope = 'Mixed_5' + chr(ord('b') + idx)
        net = block_inception_a(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points
    
      # 35 x 35 x 384
      # Reduction-A block
      net = block_reduction_a(net, 'Mixed_6a')
      if add_and_check_final('Mixed_6a', net): return net, end_points
    
      # 17 x 17 x 1024
      # 7 x Inception-B blocks
      for idx in range(7):
        block_scope = 'Mixed_6' + chr(ord('b') + idx)
        net = block_inception_b(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points
    
      # 17 x 17 x 1024
      # Reduction-B block
      net = block_reduction_b(net, 'Mixed_7a')
      if add_and_check_final('Mixed_7a', net): return net, end_points
    
      # 8 x 8 x 1536
      # 3 x Inception-C blocks
      for idx in range(3):
        block_scope = 'Mixed_7' + chr(ord('b') + idx)
        net = block_inception_c(net, block_scope)
        if add_and_check_final(block_scope, net): return net, end_points

  raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v4(inputs, num_classes=1001, is_training=True,
                 dropout_keep_prob=0.8,
                 reuse=None,
                 scope='InceptionV4',
                 create_aux_logits=True):
  """Creates the Inception V4 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxiliary logits.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped input to the logits layer
      if num_classes is 0 or None.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v4_base(inputs, scope=scope)

      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):
        # Auxiliary Head logits
        if create_aux_logits and num_classes:
          with tf.variable_scope('AuxLogits'):
            # 17 x 17 x 1024
            aux_logits = end_points['Mixed_6h']
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                         padding='VALID',
                                         scope='AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                     scope='Conv2d_1b_1x1')
            aux_logits = slim.conv2d(aux_logits, 768,
                                     aux_logits.get_shape()[1:3],
                                     padding='VALID', scope='Conv2d_2a')
            aux_logits = slim.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes,
                                              activation_fn=None,
                                              scope='Aux_logits')
            end_points['AuxLogits'] = aux_logits
    
        # Final pooling and prediction
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        with tf.variable_scope('Logits'):
          # 8 x 8 x 1536
          kernel_size = net.get_shape()[1:3]
          if kernel_size.is_fully_defined():
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a')
          else:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True,
                                 name='global_pool')
          end_points['global_pool'] = net
          if not num_classes:
            return net, end_points
          # 1 x 1 x 1536
          net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1b')
          net = slim.flatten(net, scope='PreLogitsFlatten')
          end_points['PreLogitsFlatten'] = net
          # 1536
          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    return logits, end_points

inception_v4.default_image_size = 299

```

- inception_V4网络Aux各特征图的大小
```

InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionV4/InceptionV4/Mixed_3a/concat (5, 73, 73, 160)
InceptionV4/InceptionV4/Mixed_4a/concat (5, 71, 71, 192)
InceptionV4/InceptionV4/Mixed_5a/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5b/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5c/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5d/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5e/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_6a/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6b/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6c/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6d/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6e/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6f/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6g/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6h/concat (5, 17, 17, 1024)
InceptionV4/AuxLogits/AvgPool_1a_5x5/AvgPool (5, 5, 5, 1024)
InceptionV4/AuxLogits/Conv2d_1b_1x1/Relu (5, 5, 5, 128)
InceptionV4/AuxLogits/Conv2d_2a/Relu (5, 1, 1, 768)
InceptionV4/AuxLogits/Aux_logits/BiasAdd (5, 1001)

```
- inception_V3主网络各特征图的大小
```

InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionV4/InceptionV4/Mixed_3a/concat (5, 73, 73, 160)
InceptionV4/InceptionV4/Mixed_4a/concat (5, 71, 71, 192)
InceptionV4/InceptionV4/Mixed_5a/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5b/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5c/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5d/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_5e/concat (5, 35, 35, 384)
InceptionV4/InceptionV4/Mixed_6a/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6b/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6c/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6d/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6e/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6f/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6g/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_6h/concat (5, 17, 17, 1024)
InceptionV4/InceptionV4/Mixed_7a/concat (5, 8, 8, 1536)
InceptionV4/InceptionV4/Mixed_7b/concat (5, 8, 8, 1536)
InceptionV4/InceptionV4/Mixed_7c/concat (5, 8, 8, 1536)
InceptionV4/InceptionV4/Mixed_7d/concat (5, 8, 8, 1536)
InceptionV4/Logits/AvgPool_1a/AvgPool (5, 1, 1, 1536)
InceptionV4/Logits/PreLogitsFlatten/flatten/Reshape (5, 1536)
InceptionV4/Logits/Logits/BiasAdd (5, 1001)

```
- [resnet_v1](https://github.com/Zehaos/MobileNet/blob/master/nets/resnet_v1.py)中残差结构的实现代码
```

def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride,
                             activation_fn=None, scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')
    
    output = tf.nn.relu(shortcut + residual)
    
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)

```
- resnet_v1_50网络的完整结构。50指卷积层的个数；空洞卷积(atrous convolution)
```

def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
  """Generator for v1 ResNet models.

  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
        if spatial_squeeze:
          logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(logits, scope='predictions')
        return logits, end_points
resnet_v1.default_image_size = 224

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size

```
- resnet_v1_50网络各特征图大小
```

input,(5, 224, 224, 3)
resnet_v1_50/conv1,(5, 112, 112, 64)
resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut,(5, 55, 55, 256)
resnet_v1_50/block1/unit_1/bottleneck_v1/conv1,(5, 55, 55, 64)
resnet_v1_50/block1/unit_1/bottleneck_v1/conv2,(5, 55, 55, 64)
resnet_v1_50/block1/unit_1/bottleneck_v1/conv3,(5, 55, 55, 256)
resnet_v1_50/block1/unit_1/bottleneck_v1,(5, 55, 55, 256)
resnet_v1_50/block1/unit_2/bottleneck_v1/conv1,(5, 55, 55, 64)
resnet_v1_50/block1/unit_2/bottleneck_v1/conv2,(5, 55, 55, 64)
resnet_v1_50/block1/unit_2/bottleneck_v1/conv3,(5, 55, 55, 256)
resnet_v1_50/block1/unit_2/bottleneck_v1,(5, 55, 55, 256)
resnet_v1_50/block1/unit_3/bottleneck_v1/conv1,(5, 55, 55, 64)
resnet_v1_50/block1/unit_3/bottleneck_v1/conv2,(5, 28, 28, 64)
resnet_v1_50/block1/unit_3/bottleneck_v1/conv3,(5, 28, 28, 256)
resnet_v1_50/block1/unit_3/bottleneck_v1,(5, 28, 28, 256)
resnet_v1_50/block1,(5, 28, 28, 256)
resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut,(5, 28, 28, 512)
resnet_v1_50/block2/unit_1/bottleneck_v1/conv1,(5, 28, 28, 128)
resnet_v1_50/block2/unit_1/bottleneck_v1/conv2,(5, 28, 28, 128)
resnet_v1_50/block2/unit_1/bottleneck_v1/conv3,(5, 28, 28, 512)
resnet_v1_50/block2/unit_1/bottleneck_v1,(5, 28, 28, 512)
resnet_v1_50/block2/unit_2/bottleneck_v1/conv1,(5, 28, 28, 128)
resnet_v1_50/block2/unit_2/bottleneck_v1/conv2,(5, 28, 28, 128)
resnet_v1_50/block2/unit_2/bottleneck_v1/conv3,(5, 28, 28, 512)
resnet_v1_50/block2/unit_2/bottleneck_v1,(5, 28, 28, 512)
resnet_v1_50/block2/unit_3/bottleneck_v1/conv1,(5, 28, 28, 128)
resnet_v1_50/block2/unit_3/bottleneck_v1/conv2,(5, 28, 28, 128)
resnet_v1_50/block2/unit_3/bottleneck_v1/conv3,(5, 28, 28, 512)
resnet_v1_50/block2/unit_3/bottleneck_v1,(5, 28, 28, 512)
resnet_v1_50/block2/unit_4/bottleneck_v1/conv1,(5, 28, 28, 128)
resnet_v1_50/block2/unit_4/bottleneck_v1/conv2,(5, 14, 14, 128)
resnet_v1_50/block2/unit_4/bottleneck_v1/conv3,(5, 14, 14, 512)
resnet_v1_50/block2/unit_4/bottleneck_v1,(5, 14, 14, 512)
resnet_v1_50/block2,(5, 14, 14, 512)
resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_1/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_1/bottleneck_v1/conv2,(5, 14, 14, 256)
resnet_v1_50/block3/unit_1/bottleneck_v1/conv3,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_1/bottleneck_v1,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_2/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_2/bottleneck_v1/conv2,(5, 14, 14, 256)
resnet_v1_50/block3/unit_2/bottleneck_v1/conv3,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_2/bottleneck_v1,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_3/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_3/bottleneck_v1/conv2,(5, 14, 14, 256)
resnet_v1_50/block3/unit_3/bottleneck_v1/conv3,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_3/bottleneck_v1,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_4/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_4/bottleneck_v1/conv2,(5, 14, 14, 256)
resnet_v1_50/block3/unit_4/bottleneck_v1/conv3,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_4/bottleneck_v1,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_5/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_5/bottleneck_v1/conv2,(5, 14, 14, 256)
resnet_v1_50/block3/unit_5/bottleneck_v1/conv3,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_5/bottleneck_v1,(5, 14, 14, 1024)
resnet_v1_50/block3/unit_6/bottleneck_v1/conv1,(5, 14, 14, 256)
resnet_v1_50/block3/unit_6/bottleneck_v1/conv2,(5, 7, 7, 256)
resnet_v1_50/block3/unit_6/bottleneck_v1/conv3,(5, 7, 7, 1024)
resnet_v1_50/block3/unit_6/bottleneck_v1,(5, 7, 7, 1024)
resnet_v1_50/block3,(5, 7, 7, 1024)
resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_1/bottleneck_v1/conv1,(5, 7, 7, 512)
resnet_v1_50/block4/unit_1/bottleneck_v1/conv2,(5, 7, 7, 512)
resnet_v1_50/block4/unit_1/bottleneck_v1/conv3,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_1/bottleneck_v1,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_2/bottleneck_v1/conv1,(5, 7, 7, 512)
resnet_v1_50/block4/unit_2/bottleneck_v1/conv2,(5, 7, 7, 512)
resnet_v1_50/block4/unit_2/bottleneck_v1/conv3,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_2/bottleneck_v1,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_3/bottleneck_v1/conv1,(5, 7, 7, 512)
resnet_v1_50/block4/unit_3/bottleneck_v1/conv2,(5, 7, 7, 512)
resnet_v1_50/block4/unit_3/bottleneck_v1/conv3,(5, 7, 7, 2048)
resnet_v1_50/block4/unit_3/bottleneck_v1,(5, 7, 7, 2048)
resnet_v1_50/block4,(5, 7, 7, 2048)
resnet_v1_50/logits,(5, 1, 1, 1000)
predictions,(5, 1000)

```
- resnet_v2网络中残差结构的实现代码。每个block首次卷积前进行批归一化
```

def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')
    
    output = shortcut + residual
    
    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)

```
- [resenet_v2_50](https://github.com/Zehaos/MobileNet/blob/master/nets/resnet_v2.py)网络的完整结构
```

def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
  """Generator for v2 (preactivation) ResNet models.

  This function generates a family of ResNet v2 models. See the resnet_v2_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.

  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.

  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether is training or not.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it. If excluded, `inputs` should be the
      results of an activation-less convolution.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits')
        if spatial_squeeze:
          logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(logits, scope='predictions')
        return logits, end_points
resnet_v2.default_image_size = 224

def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', bottleneck, [(2048, 512, 1)] * 3)]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size

```
- resnet_v2_50网络的各层特征图大小
```

input,(5, 224, 224, 3)
resnet_v2_50/conv1,(5, 112, 112, 64)
resnet_v2_50/block1/unit_1/bottleneck_v2/shortcut,(5, 55, 55, 256)
resnet_v2_50/block1/unit_1/bottleneck_v2/conv1,(5, 55, 55, 64)
resnet_v2_50/block1/unit_1/bottleneck_v2/conv2,(5, 55, 55, 64)
resnet_v2_50/block1/unit_1/bottleneck_v2/conv3,(5, 55, 55, 256)
resnet_v2_50/block1/unit_1/bottleneck_v2,(5, 55, 55, 256)
resnet_v2_50/block1/unit_2/bottleneck_v2/conv1,(5, 55, 55, 64)
resnet_v2_50/block1/unit_2/bottleneck_v2/conv2,(5, 55, 55, 64)
resnet_v2_50/block1/unit_2/bottleneck_v2/conv3,(5, 55, 55, 256)
resnet_v2_50/block1/unit_2/bottleneck_v2,(5, 55, 55, 256)
resnet_v2_50/block1/unit_3/bottleneck_v2/conv1,(5, 55, 55, 64)
resnet_v2_50/block1/unit_3/bottleneck_v2/conv2,(5, 28, 28, 64)
resnet_v2_50/block1/unit_3/bottleneck_v2/conv3,(5, 28, 28, 256)
resnet_v2_50/block1/unit_3/bottleneck_v2,(5, 28, 28, 256)
resnet_v2_50/block1,(5, 28, 28, 256)
resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut,(5, 28, 28, 512)
resnet_v2_50/block2/unit_1/bottleneck_v2/conv1,(5, 28, 28, 128)
resnet_v2_50/block2/unit_1/bottleneck_v2/conv2,(5, 28, 28, 128)
resnet_v2_50/block2/unit_1/bottleneck_v2/conv3,(5, 28, 28, 512)
resnet_v2_50/block2/unit_1/bottleneck_v2,(5, 28, 28, 512)
resnet_v2_50/block2/unit_2/bottleneck_v2/conv1,(5, 28, 28, 128)
resnet_v2_50/block2/unit_2/bottleneck_v2/conv2,(5, 28, 28, 128)
resnet_v2_50/block2/unit_2/bottleneck_v2/conv3,(5, 28, 28, 512)
resnet_v2_50/block2/unit_2/bottleneck_v2,(5, 28, 28, 512)
resnet_v2_50/block2/unit_3/bottleneck_v2/conv1,(5, 28, 28, 128)
resnet_v2_50/block2/unit_3/bottleneck_v2/conv2,(5, 28, 28, 128)
resnet_v2_50/block2/unit_3/bottleneck_v2/conv3,(5, 28, 28, 512)
resnet_v2_50/block2/unit_3/bottleneck_v2,(5, 28, 28, 512)
resnet_v2_50/block2/unit_4/bottleneck_v2/conv1,(5, 28, 28, 128)
resnet_v2_50/block2/unit_4/bottleneck_v2/conv2,(5, 14, 14, 128)
resnet_v2_50/block2/unit_4/bottleneck_v2/conv3,(5, 14, 14, 512)
resnet_v2_50/block2/unit_4/bottleneck_v2,(5, 14, 14, 512)
resnet_v2_50/block2,(5, 14, 14, 512)
resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_1/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_1/bottleneck_v2/conv2,(5, 14, 14, 256)
resnet_v2_50/block3/unit_1/bottleneck_v2/conv3,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_1/bottleneck_v2,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_2/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_2/bottleneck_v2/conv2,(5, 14, 14, 256)
resnet_v2_50/block3/unit_2/bottleneck_v2/conv3,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_2/bottleneck_v2,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_3/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_3/bottleneck_v2/conv2,(5, 14, 14, 256)
resnet_v2_50/block3/unit_3/bottleneck_v2/conv3,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_3/bottleneck_v2,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_4/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_4/bottleneck_v2/conv2,(5, 14, 14, 256)
resnet_v2_50/block3/unit_4/bottleneck_v2/conv3,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_4/bottleneck_v2,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_5/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_5/bottleneck_v2/conv2,(5, 14, 14, 256)
resnet_v2_50/block3/unit_5/bottleneck_v2/conv3,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_5/bottleneck_v2,(5, 14, 14, 1024)
resnet_v2_50/block3/unit_6/bottleneck_v2/conv1,(5, 14, 14, 256)
resnet_v2_50/block3/unit_6/bottleneck_v2/conv2,(5, 7, 7, 256)
resnet_v2_50/block3/unit_6/bottleneck_v2/conv3,(5, 7, 7, 1024)
resnet_v2_50/block3/unit_6/bottleneck_v2,(5, 7, 7, 1024)
resnet_v2_50/block3,(5, 7, 7, 1024)
resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_1/bottleneck_v2/conv1,(5, 7, 7, 512)
resnet_v2_50/block4/unit_1/bottleneck_v2/conv2,(5, 7, 7, 512)
resnet_v2_50/block4/unit_1/bottleneck_v2/conv3,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_1/bottleneck_v2,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_2/bottleneck_v2/conv1,(5, 7, 7, 512)
resnet_v2_50/block4/unit_2/bottleneck_v2/conv2,(5, 7, 7, 512)
resnet_v2_50/block4/unit_2/bottleneck_v2/conv3,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_2/bottleneck_v2,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_3/bottleneck_v2/conv1,(5, 7, 7, 512)
resnet_v2_50/block4/unit_3/bottleneck_v2/conv2,(5, 7, 7, 512)
resnet_v2_50/block4/unit_3/bottleneck_v2/conv3,(5, 7, 7, 2048)
resnet_v2_50/block4/unit_3/bottleneck_v2,(5, 7, 7, 2048)
resnet_v2_50/block4,(5, 7, 7, 2048)
resnet_v2_50/logits,(5, 1, 1, 1000)
predictions,(5, 1000)

```
- inception_resnet结构的[v2](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_resnet_v2.py)版本中Block35层,inception结构的残差。
```

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)

  return net

```
- inception_resnet_v2的完整结构
```

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

    net += scaled_up
    if activation_fn:
      net = activation_fn(net)

  return net

def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
    
    net += scaled_up
    if activation_fn:
      net = activation_fn(net)

  return net

def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')

    scaled_up = up * scale
    if activation_fn == tf.nn.relu6:
      # Use clip_by_value to simulate bandpass activation.
      scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)
    
    net += scaled_up
    if activation_fn:
      net = activation_fn(net)

  return net

def inception_resnet_v2_base(inputs,
                             final_endpoint='Conv2d_7b_1x1',
                             output_stride=16,
                             align_feature_maps=False,
                             scope=None,
                             activation_fn=tf.nn.relu):
  """Inception model from  http://arxiv.org/abs/1602.07261.

  Constructs an Inception Resnet v2 network from inputs to the given final
  endpoint. This method can construct the network up to the final inception
  block Conv2d_7b_1x1.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
    output_stride: A scalar that specifies the requested ratio of input to
      output spatial resolution. Only supports 8 and 16.
    align_feature_maps: When true, changes all the VALID paddings in the network
      to SAME padding so that the feature maps are aligned.
    scope: Optional variable_scope.
    activation_fn: Activation function for block scopes.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
      or if the output_stride is not 8 or 16, or if the output_stride is 8 and
      we request an end point after 'PreAuxLogits'.
  """
  if output_stride != 8 and output_stride != 16:
    raise ValueError('output_stride must be 8 or 16.')

  padding = 'SAME' if align_feature_maps else 'VALID'

  end_points = {}

  def add_and_check_final(name, net):
    end_points[name] = net
    return name == final_endpoint

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # 149 x 149 x 32
      net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding,
                        scope='Conv2d_1a_3x3')
      if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

      # 147 x 147 x 32
      net = slim.conv2d(net, 32, 3, padding=padding,
                        scope='Conv2d_2a_3x3')
      if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
      # 147 x 147 x 64
      net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
      if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
      # 73 x 73 x 64
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_3a_3x3')
      if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
      # 73 x 73 x 80
      net = slim.conv2d(net, 80, 1, padding=padding,
                        scope='Conv2d_3b_1x1')
      if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
      # 71 x 71 x 192
      net = slim.conv2d(net, 192, 3, padding=padding,
                        scope='Conv2d_4a_3x3')
      if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
      # 35 x 35 x 192
      net = slim.max_pool2d(net, 3, stride=2, padding=padding,
                            scope='MaxPool_5a_3x3')
      if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points
    
      # 35 x 35 x 320
      with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                      scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                      scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                       scope='AvgPool_0a_3x3')
          tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                     scope='Conv2d_0b_1x1')
        net = tf.concat(
            [tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)
    
      if add_and_check_final('Mixed_5b', net): return net, end_points
      # TODO(alemi): Register intermediate endpoints
      net = slim.repeat(net, 10, block35, scale=0.17,
                        activation_fn=activation_fn)
    
      # 17 x 17 x 1088 if output_stride == 8,
      # 33 x 33 x 1088 if output_stride == 16
      use_atrous = output_stride == 8
    
      with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                   padding=padding,
                                   scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                      stride=1 if use_atrous else 2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    
      if add_and_check_final('Mixed_6a', net): return net, end_points
    
      # TODO(alemi): register intermediate endpoints
      with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
        net = slim.repeat(net, 20, block17, scale=0.10,
                          activation_fn=activation_fn)
      if add_and_check_final('PreAuxLogits', net): return net, end_points
    
      if output_stride == 8:
        # TODO(gpapan): Properly support output_stride for the rest of the net.
        raise ValueError('output_stride==8 is only supported up to the '
                         'PreAuxlogits end_point for now.')
    
      # 8 x 8 x 2080
      with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                     padding=padding,
                                     scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                      scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                      padding=padding,
                                      scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_3'):
          tower_pool = slim.max_pool2d(net, 3, stride=2,
                                       padding=padding,
                                       scope='MaxPool_1a_3x3')
        net = tf.concat(
            [tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
    
      if add_and_check_final('Mixed_7a', net): return net, end_points
    
      # TODO(alemi): register intermediate endpoints
      net = slim.repeat(net, 9, block8, scale=0.20, activation_fn=activation_fn)
      net = block8(net, activation_fn=None)
    
      # 8 x 8 x 1536
      net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
      if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points
    
    raise ValueError('final_endpoint (%s) not recognized', final_endpoint)

def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2',
                        create_aux_logits=True,
                        activation_fn=tf.nn.relu):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
      Dimension batch_size may be undefined. If create_aux_logits is false,
      also height and width may be undefined.
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before  dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.
    activation_fn: Activation function for conv2d.

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):

      net, end_points = inception_resnet_v2_base(inputs, scope=scope,
                                                 activation_fn=activation_fn)
    
      if create_aux_logits and num_classes:
        with tf.variable_scope('AuxLogits'):
          aux = end_points['PreAuxLogits']
          aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID',
                                scope='Conv2d_1a_3x3')
          aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
          aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                            padding='VALID', scope='Conv2d_2a_5x5')
          aux = slim.flatten(aux)
          aux = slim.fully_connected(aux, num_classes, activation_fn=None,
                                     scope='Logits')
          end_points['AuxLogits'] = aux
    
      with tf.variable_scope('Logits'):
        # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
        # can be set to False to disable pooling here (as in resnet_*()).
        kernel_size = net.get_shape()[1:3]
        if kernel_size.is_fully_defined():
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_8x8')
        else:
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
        if not num_classes:
          return net, end_points
        net = slim.flatten(net)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')
        end_points['PreLogitsFlatten'] = net
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='Logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    
    return logits, end_points

inception_resnet_v2.default_image_size = 299

```
- inception_resnetV2网络Aux各特征图大小
```

InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionResnetV2/InceptionResnetV2/MaxPool_3a_3x3/MaxPool (5, 73, 73, 64)
InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/Relu (5, 73, 73, 80)
InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu (5, 71, 71, 192)
InceptionResnetV2/InceptionResnetV2/MaxPool_5a_3x3/MaxPool (5, 35, 35, 192)
InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat (5, 35, 35, 320)
InceptionResnetV2/InceptionResnetV2/Mixed_6a/concat (5, 17, 17, 1088)
InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu (5, 17, 17, 1088)
InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu (5, 8, 8, 1536)
InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu (5, 8, 8, 1536)
InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu (5, 8, 8, 1536)
InceptionResnetV2/AuxLogits/Logits/BiasAdd (5, 1001)

```
- inception_resnetV2主网络各特征图的大小
```

InceptionResnetV2/InceptionResnetV2/Conv2d_1a_3x3/Relu (5, 149, 149, 32)
InceptionResnetV2/InceptionResnetV2/Conv2d_2a_3x3/Relu (5, 147, 147, 32)
InceptionResnetV2/InceptionResnetV2/Conv2d_2b_3x3/Relu (5, 147, 147, 64)
InceptionResnetV2/InceptionResnetV2/MaxPool_3a_3x3/MaxPool (5, 73, 73, 64)
InceptionResnetV2/InceptionResnetV2/Conv2d_3b_1x1/Relu (5, 73, 73, 80)
InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu (5, 71, 71, 192)
InceptionResnetV2/InceptionResnetV2/MaxPool_5a_3x3/MaxPool (5, 35, 35, 192)
InceptionResnetV2/InceptionResnetV2/Mixed_5b/concat (5, 35, 35, 320)
InceptionResnetV2/InceptionResnetV2/Mixed_6a/concat (5, 17, 17, 1088)
InceptionResnetV2/InceptionResnetV2/Repeat_1/block17_20/Relu (5, 17, 17, 1088)
InceptionResnetV2/InceptionResnetV2/Mixed_7a/concat (5, 8, 8, 2080)
InceptionResnetV2/InceptionResnetV2/Conv2d_7b_1x1/Relu (5, 8, 8, 1536)
InceptionResnetV2/Logits/AvgPool_1a_8x8/AvgPool (5, 1, 1, 1536)
InceptionResnetV2/Logits/Dropout/dropout_1/mul_1 (5, 1536)
InceptionResnetV2/Logits/Logits/BiasAdd (5, 1001)

```
- [谷歌Inception网络中的Inception-V3到Inception-V4具体作了哪些优化？](https://www.zhihu.com/question/50370954)
- [Mobilenet v1](https://github.com/Zehaos/MobileNet/blob/master/nets/mobilenet.py)网络结构。depthwise_separable_conv(深度可分离卷积)
```

def mobilenet(inputs,
          num_classes=1000,
          is_training=True,
          width_multiplier=1,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """

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

  with tf.variable_scope(scope) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                        activation_fn=None,
                        outputs_collections=[end_points_collection]):
      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu,
                          fused=True):
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME', scope='conv_1')
        net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 64, width_multiplier, sc='conv_ds_2')
        net = _depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
        net = _depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
        net = _depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
        net = _depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
        net = _depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
        net = _depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')
    
        net = _depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
        net = _depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
        net = slim.avg_pool2d(net, [7, 7], scope='avg_pool_15')
    
    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
    end_points['squeeze'] = net
    logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc_16')
    predictions = slim.softmax(logits, scope='Predictions')
    
    end_points['Logits'] = logits
    end_points['Predictions'] = predictions

  return logits, end_points

mobilenet.default_image_size = 224

```
- mobilenet网络各层特征图大小
```

input,(5, 224, 224, 3)
MobileNet/conv_1,(5, 112, 112, 32)
MobileNet/conv_ds_2/depthwise_conv,(5, 112, 112, 32)
MobileNet/conv_ds_2/pointwise_conv,(5, 112, 112, 64)
MobileNet/conv_ds_3/depthwise_conv,(5, 56, 56, 64)
MobileNet/conv_ds_3/pointwise_conv,(5, 56, 56, 128)
MobileNet/conv_ds_4/depthwise_conv,(5, 56, 56, 128)
MobileNet/conv_ds_4/pointwise_conv,(5, 56, 56, 128)
MobileNet/conv_ds_5/depthwise_conv,(5, 28, 28, 128)
MobileNet/conv_ds_5/pointwise_conv,(5, 28, 28, 256)
MobileNet/conv_ds_6/depthwise_conv,(5, 28, 28, 256)
MobileNet/conv_ds_6/pointwise_conv,(5, 28, 28, 256)
MobileNet/conv_ds_7/depthwise_conv,(5, 14, 14, 256)
MobileNet/conv_ds_7/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_8/depthwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_8/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_9/depthwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_9/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_10/depthwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_10/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_11/depthwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_11/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_12/depthwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_12/pointwise_conv,(5, 14, 14, 512)
MobileNet/conv_ds_13/depthwise_conv,(5, 7, 7, 512)
MobileNet/conv_ds_13/pointwise_conv,(5, 7, 7, 1024)
MobileNet/conv_ds_14/depthwise_conv,(5, 7, 7, 1024)
MobileNet/conv_ds_14/pointwise_conv,(5, 7, 7, 1024)
squeeze,(5, 1024)
Logits,(5, 1000)
Predictions,(5, 1000)

```
- [fcn](https://github.com/shekkizh/FCN.tensorflow/blob/master/FCN.py)
```

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    
    return net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    
    weights = np.squeeze(model_data['layers'])
    
    processed_image = utils.process_image(image, mean_pixel)
    
    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
    
        pool5 = utils.max_pool_2x2(conv_final_layer)
    
        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)
    
        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)
    
        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
    
        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")
    
        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")
    
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)
    
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
    
    return tf.expand_dims(annotation_pred, dim=3), conv_t3

```
- 网络各特征图大小, 输出通道151表示类别数
```

name: input, shape:(5, 224, 224, 3)
name: conv1_1, shape: (5, 224, 224, 64)
name: relu1_1, shape: (5, 224, 224, 64)
name: conv1_2, shape: (5, 224, 224, 64)
name: relu1_2, shape: (5, 224, 224, 64)
name: pool1, shape: (5, 112, 112, 64)
name: conv2_1, shape: (5, 112, 112, 128)
name: relu2_1, shape: (5, 112, 112, 128)
name: conv2_2, shape: (5, 112, 112, 128)
name: relu2_2, shape: (5, 112, 112, 128)
name: pool2, shape: (5, 56, 56, 128)
name: conv3_1, shape: (5, 56, 56, 256)
name: relu3_1, shape: (5, 56, 56, 256)
name: conv3_2, shape: (5, 56, 56, 256)
name: relu3_2, shape: (5, 56, 56, 256)
name: conv3_3, shape: (5, 56, 56, 256)
name: relu3_3, shape: (5, 56, 56, 256)
name: conv3_4, shape: (5, 56, 56, 256)
name: relu3_4, shape: (5, 56, 56, 256)
name: pool3, shape: (5, 28, 28, 256)
name: conv4_1, shape: (5, 28, 28, 512)
name: relu4_1, shape: (5, 28, 28, 512)
name: conv4_2, shape: (5, 28, 28, 512)
name: relu4_2, shape: (5, 28, 28, 512)
name: conv4_3, shape: (5, 28, 28, 512)
name: relu4_3, shape: (5, 28, 28, 512)
name: conv4_4, shape: (5, 28, 28, 512)
name: relu4_4, shape: (5, 28, 28, 512)
name: pool4, shape: (5, 14, 14, 512)
name: conv5_1, shape: (5, 14, 14, 512)
name: relu5_1, shape: (5, 14, 14, 512)
name: conv5_2, shape: (5, 14, 14, 512)
name: relu5_2, shape: (5, 14, 14, 512)
name: conv5_3, shape: (5, 14, 14, 512)
name: relu5_3, shape: (5, 14, 14, 512)
name: conv5_4, shape: (5, 14, 14, 512)
name: relu5_4, shape: (5, 14, 14, 512)
name: conv6, shape: (5, 7, 7, 4096)
name: conv7, shape: (5, 7, 7, 4096)
name: conv8, shape: (5, 7, 7, 151)
name: fuse_1, shape: (5, 14, 14, 512)
name: conv_t2, shape: (5, 28, 28, 256)
name: fuse_2, shape: (5, 28, 28, 256)
name: logits, shape: (5, 224, 224, 151)

```
- 输入图像分辨率改变时的各特征图大小
```

name: input, shape: (5, 336, 223, 3)
name: conv1_1, shape: (5, 336, 223, 64)
name: relu1_1, shape: (5, 336, 223, 64)
name: conv1_2, shape: (5, 336, 223, 64)
name: relu1_2, shape: (5, 336, 223, 64)
name: pool1, shape: (5, 168, 112, 64)
name: conv2_1, shape: (5, 168, 112, 128)
name: relu2_1, shape: (5, 168, 112, 128)
name: conv2_2, shape: (5, 168, 112, 128)
name: relu2_2, shape: (5, 168, 112, 128)
name: pool2, shape: (5, 84, 56, 128)
name: conv3_1, shape: (5, 84, 56, 256)
name: relu3_1, shape: (5, 84, 56, 256)
name: conv3_2, shape: (5, 84, 56, 256)
name: relu3_2, shape: (5, 84, 56, 256)
name: conv3_3, shape: (5, 84, 56, 256)
name: relu3_3, shape: (5, 84, 56, 256)
name: conv3_4, shape: (5, 84, 56, 256)
name: relu3_4, shape: (5, 84, 56, 256)
name: pool3, shape: (5, 42, 28, 256)
name: conv4_1, shape: (5, 42, 28, 512)
name: relu4_1, shape: (5, 42, 28, 512)
name: conv4_2, shape: (5, 42, 28, 512)
name: relu4_2, shape: (5, 42, 28, 512)
name: conv4_3, shape: (5, 42, 28, 512)
name: relu4_3, shape: (5, 42, 28, 512)
name: conv4_4, shape: (5, 42, 28, 512)
name: relu4_4, shape: (5, 42, 28, 512)
name: pool4, shape: (5, 21, 14, 512)
name: conv5_1, shape: (5, 21, 14, 512)
name: relu5_1, shape: (5, 21, 14, 512)
name: conv5_2, shape: (5, 21, 14, 512)
name: relu5_2, shape: (5, 21, 14, 512)
name: conv5_3, shape: (5, 21, 14, 512)
name: relu5_3, shape: (5, 21, 14, 512)
name: conv5_4, shape: (5, 21, 14, 512)
name: relu5_4, shape: (5, 21, 14, 512)
name: conv6, shape: (5, 11, 7, 4096)
name: conv7, shape: (5, 11, 7, 4096)
name: conv8, shape: (5, 11, 7, 151)
name: fuse_1, shape: (5, 21, 14, 512)
name: conv_t2, shape: (5, 42, 28, 256)
name: fuse_2, shape: (5, 42, 28, 256)
name: logits, shape: (5, 336, 223, 151)

```

```
