- [Alexnet](https://github.com/Zehaos/MobileNet/blob/master/nets/alexnet.py)全卷积结构
- 


```

	with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
	                    outputs_collections=[end_points_collection]):
	  net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
	                    scope='conv1')
	  net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
	  net = slim.conv2d(net, 192, [5, 5], scope='conv2')
	  net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
	  net = slim.conv2d(net, 384, [3, 3], scope='conv3')
	  net = slim.conv2d(net, 384, [3, 3], scope='conv4')
	  net = slim.conv2d(net, 256, [3, 3], scope='conv5')
	  net = slim.max_pool2d(net, [3, 3], 2, scope='pool5') #注意pool5操作的卷积核大小为3，所以得到的特征图为(26-3)/2+1=12
	
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
   