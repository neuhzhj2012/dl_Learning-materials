- [lenet](https://github.com/sujaybabruwad/LeNet-in-Tensorflow/blob/master/LeNet-Lab.ipynb)tf基础函数版

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
- 

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
- 

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
- 


```

	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
	                    outputs_collections=[end_points_collection]):
	  net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
	                    scope='conv1')
	  net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
	  net = slim.conv2d(net, 192, [5, 5], scope='conv2')
	  net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')#注意pool2操作的卷积核大小为3，所以得到的特征图为(26-3)/2+1=12
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


- inception结构，展示代码为[v1](https://github.com/Zehaos/MobileNet/blob/master/nets/inception_v1.py)结构的Mixed_3b层
-


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
   