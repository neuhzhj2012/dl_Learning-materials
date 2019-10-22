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
- inception结构，展示代码为[v1](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v1.py)结构的Mixed_3b层
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
- inception结构的[v2](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v2.py)版本中Mixed_3b层
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

- inception_V2的完整结构，可分离卷积
	
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
- inception结构[V3](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v3.py)版本中Mixed_6b层，将7*7卷积核拆分为1*7和7*1的两层卷积达到相同感受野;输入图像分辨率有224*224变为299*299;同时训练两个不同深度的网络

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