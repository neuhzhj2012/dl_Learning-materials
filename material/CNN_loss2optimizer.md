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
- [slim LOSS](https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py)该模块为3个小部分：训练函数(包括loss和optimizer)、数据加载过程和网络加载过程。
1. 训练函数(loss和optimizer)

```
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset

      train_image_size = FLAGS.train_image_size or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4,
            scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None

    if FLAGS.quantize_delay >= 0:
      tf.contrib.quantize.create_training_graph(
          quant_delay=FLAGS.quantize_delay)

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_dir,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        sync_optimizer=optimizer if FLAGS.sync_replicas else None)
```
如上训练函数中，通过train_tensor定位到
```
total_loss, clones_gradients = model_deploy.optimize_clones(
    clones,
    optimizer,
    var_list=variables_to_train)
```
进而通过[model_deploy.py](https://github.com/tensorflow/models/blob/master/research/slim/deployment/model_deploy.py)文件得到损失值的
```
def _gather_clone_loss(clone, num_clones, regularization_losses):
  """Gather the loss for a single clone.
  Args:
    clone: A Clone namedtuple.
    num_clones: The number of clones being deployed.
    regularization_losses: Possibly empty list of regularization_losses
      to add to the clone losses.
  Returns:
    A tensor for the total loss for the clone.  Can be None.
  """
  # The return value.
  sum_loss = None
  # Individual components of the loss that will need summaries.
  clone_loss = None
  regularization_loss = None
  # Compute and aggregate losses on the clone device.
  with tf.device(clone.device):
    all_losses = []
    clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
    if clone_losses:
      clone_loss = tf.add_n(clone_losses, name='clone_loss')
      if num_clones > 1:
        clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                            name='scaled_clone_loss')
      all_losses.append(clone_loss)
    if regularization_losses:
      regularization_loss = tf.add_n(regularization_losses,
                                     name='regularization_loss')
      all_losses.append(regularization_loss)
    if all_losses:
      sum_loss = tf.add_n(all_losses)
  # Add the summaries out of the clone device block.
  if clone_loss is not None:
    tf.summary.scalar('/'.join(filter(None,
                                      ['Losses', clone.scope, 'clone_loss'])),
                      clone_loss)
  if regularization_loss is not None:
    tf.summary.scalar('Losses/regularization_loss', regularization_loss)
  return sum_loss
```
那么clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)中的loss又是如何得到的呢，这得回到训练函数中slim.losses.softmax_cross_entropy，可通过[该函数](https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/contrib/losses/python/losses/loss_ops.py)的定义中进行追踪softmax_cross_entropy->compute_weighted_loss->add_loss。至此，当前batch图像前向计算得到的损失值加入tf.GraphKeys.LOSSES（'losses'）中。
```
def add_loss(loss, loss_collection=ops.GraphKeys.LOSSES):
  """Adds a externally defined loss to the collection of losses.
  Args:
    loss: A loss `Tensor`.
    loss_collection: Optional collection to add the loss to.
  """
  if loss_collection:
    ops.add_to_collection(loss_collection, loss)
```
优化器通过[训练参数的optimizer字段](https://github.com/tensorflow/models/blob/master/research/slim/train_image_classifier.py)进行选择。
```
def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer
```
2. slim的数据加载过程,其中包括[Dataset类](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/slim/python/slim/data/dataset.py),[DatasetDataProvider类](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py)。数据加载过程[imagenet.py](https://github.com/tensorflow/models/blob/ba87e2c6afd383a962f805b290efb0218068d096/research/slim/datasets/imagenet.py)，其中[TfExampleDecoder](https://github.com/tensorflow/models/blob/fe748d4a4a1576b57c279014ac0ceb47344399c4/research/object_detection/data_decoders/tf_example_decoder.py)负责图片和类别对应的具体解析；
label文件读取[程序dataset_utils.py](https://github.com/tensorflow/models/blob/master/research/slim/datasets/dataset_utils.py)
```
def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading ImageNet.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  if split_name not in _SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/class/label': tf.FixedLenFeature(
          [], dtype=tf.int64, default_value=-1),
      'image/class/text': tf.FixedLenFeature(
          [], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/ymin': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/xmax': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/bbox/ymax': tf.VarLenFeature(
          dtype=tf.float32),
      'image/object/class/label': tf.VarLenFeature(
          dtype=tf.int64),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
      'object/bbox': slim.tfexample_decoder.BoundingBox(
          ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
      'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if LOAD_READABLE_NAMES:
    if dataset_utils.has_labels(dataset_dir):
      labels_to_names = dataset_utils.read_label_file(dataset_dir)
    else:
      labels_to_names = create_readable_names_for_imagenet_labels()
      dataset_utils.write_label_file(labels_to_names, dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=_SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
```
通过tf_record的生成代码[build_image_data.py](https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py)方便理解数据的存储方式。
```
def _convert_to_example(filename, image_buffer, label, text, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
      'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
      'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
      'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example
```
3. 网络的加载过程是通过[nets_factory.py](https://github.com/tensorflow/models/blob/377c5285db605ebd369222b815b86c81e33d1c41/research/slim/nets/nets_factory.py)函数实现的。[functools.wraps旨在消除装饰器对原函数造成的影响](https://zhuanlan.zhihu.com/p/45535784)，即对原函数的相关属性进行拷贝，已达到装饰器不修改原函数的目的。

```
def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
  """Returns a network_fn such as `logits, end_points = network_fn(images)`.
  Args:
    name: The name of the network.
    num_classes: The number of classes to use for classification. If 0 or None,
      the logits layer is omitted and its input features are returned instead.
    weight_decay: The l2 coefficient for the model weights.
    is_training: `True` if the model is being used for training and `False`
      otherwise.
  Returns:
    network_fn: A function that applies the model to a batch of images. It has
      the following signature:
          net, end_points = network_fn(images)
      The `images` input is a tensor of shape [batch_size, height, width, 3 or
       1] with height = width = network_fn.default_image_size. (The
      permissibility and treatment of other sizes depends on the network_fn.)
      The returned `end_points` are a dictionary of intermediate activations.
      The returned `net` is the topmost layer, depending on `num_classes`:
      If `num_classes` was a non-zero integer, `net` is a logits tensor
      of shape [batch_size, num_classes].
      If `num_classes` was 0 or `None`, `net` is a tensor with the input
      to the logits layer of shape [batch_size, 1, 1, num_features] or
      [batch_size, num_features]. Dropout has not been applied to this
      (even if the network's original classification does); it remains for
      the caller to do this or not.
  Raises:
    ValueError: If network `name` is not recognized.
  """
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map[name]
  @functools.wraps(func)
  def network_fn(images, **kwargs):
    arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes=num_classes, is_training=is_training,
                  **kwargs)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn
```



