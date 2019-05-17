## [生活不只是工作](https://mp.weixin.qq.com/s/pwbOv22KFF-f2BdJ6EA7rA)
## 其他
- [Git使用教程](https://mp.weixin.qq.com/s/5zW3_De8fseRXSiudfDJ4w)
- [一周的机器学习](https://mp.weixin.qq.com/s/aC7C8Zi9tT7h3e3TqbtYtA)
- “[软化](https://mp.weixin.qq.com/s/Duouc-ErqGqO4aTNA0NiyA)”就是把一些本来不可导的函数用一些可导函数来近似
- [标注工具](https://mp.weixin.qq.com/s/oWnvBEuVt6AQTdvdt11AeQ)
- [code review](https://mp.weixin.qq.com/s/U3x15KkJTaFis6Bh7yHFyw)
- [大数据，云计算，AI讲解](https://mp.weixin.qq.com/s/45YVq5TlhJ20asRbJrujLA)
- [知识图谱](https://mp.weixin.qq.com/s/fuI9U7aZpuk-WX6GQNtOuA)
- [最通俗的 Python3 网络爬虫入门](https://mp.weixin.qq.com/s/je7w3IgchDTpXADYAbh-pg)
## 概念
### 训练参数
- one epoch = one forward pass and one backward pass of all the training examples
- batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
- number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

```
Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.  

The term "batch" is ambiguous: some people use it to designate the entire training set, and some people use it to refer to the number of training examples in one forward/backward pass (as I did in this answer). To avoid that ambiguity and make clear that batch corresponds to the number of training examples in one forward/backward pass, one can use the term mini-batch.
```





## 人工智能
- [介绍](https://zhuanlan.zhihu.com/p/31650418)
## 机器学习
- [十大机器学习算法](https://mp.weixin.qq.com/s/gx7rIedkengj54wrj-JGZw)

## 深度学习
- [十大深度学习方法](https://mp.weixin.qq.com/s/V7WTcs_yi9qDil3Z1vjwvg)
- 深度学习[代码发展历程](https://mp.weixin.qq.com/s/htNUr1_NfMODj3t0VH5tzQ)
- [强化学习的 18 个关键问题](https://mp.weixin.qq.com/s/I8IwPCY6-zocJKFXMr6rUg)
- [18中热门GAN,附代码](http://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247497494&idx=6&sn=a1956065373bfae8ac12463be930cab4&chksm=e8d04064dfa7c9720779aaf3b6224b408133e285be383b90ef4c88a572661c80f5223b737eb2&mpshare=1&scene=23&srcid=0501ijXn9ieadUdRxahvFKy1#rd)
### 新概念
- [图网络GNN](https://mp.weixin.qq.com/s/yYJw7gvploiRCTfwmeSwZQ)
### 入门教程

- [卷积：如何成为一个很厉害的神经网络](https://jizhi.im/blog/post/intuitive_explanation_cnn)
- [看图说话](https://mp.weixin.qq.com/s/-b8FuEQlpEb5G0L0QAxVEA)
- [tf分布式训练](https://mp.weixin.qq.com/s/DAV3TDI4JYr0sXqTGU6t2A)
- [NLP](https://mp.weixin.qq.com/s/Gq7aV0Lx5fQZWXxgod2PlA)
- [NLP基础资源](https://zhuanlan.zhihu.com/p/31031566)
- [自动驾驶](https://zhuanlan.zhihu.com/p/31119925)
- [语义分割](https://mp.weixin.qq.com/s/Amr34SdrPZho1GQpFS7WBA)
- [GAN入门1](https://zhuanlan.zhihu.com/p/24767059)
- [GAN入门2](https://www.msra.cn/zh-cn/news/features/gan-20170511)
- [WGAN](https://zhuanlan.zhihu.com/p/25071913)
- [强化学习](https://mp.weixin.qq.com/s/gFHbLF-q91sddMAX1CRbEQ)
- [一文了解强化学习](https://mp.weixin.qq.com/s/EA3rkdht4tC-WAlk7-4nyw)
- [MTL](https://github.com/jg8610/multi-task-part-1-notebook/tree/master)(Multi-task learning)
- [多标签分类](http://www.atyun.com/5376.html)
- [StartGan风格迁移](https://mp.weixin.qq.com/s/DYSnAwP9xt-p0ihsEtKm1Q)
- [重现“世界模型”实验，无监督方式快速训练](https://mp.weixin.qq.com/s/GHjmiB6F2W3Zo8gVllTyyQ)


```
multi-label cross-entropy loss；  
弱监督学习；  
去除图片最强判别力的区域后继续训练分类网络，网络得到新的较强判别力区域
```

- [tf-lite在安卓上的部署](https://mp.weixin.qq.com/s/Ks4-s4D40eElY8-903JRZQ)
- [知识图谱从0到1](https://mp.weixin.qq.com/s/Lg86oFwJbd1uskZtsiQ3UQ)
### 教程
- [touch六个教程](https://zhuanlan.zhihu.com/p/32183361)
### 网络可视化
- [ConvNetJS](http://cs.stanford.edu/people/karpathy/convnetjs/index.html)
### 应用教程
- [GAN实现阴影检测和去除](https://mp.weixin.qq.com/s/hZa_ctDNgf33YuSafULxiQ)
- [语义分割中的弱监督学习](https://mp.weixin.qq.com/s/Amr34SdrPZho1GQpFS7WBA)
- [使用注意力机制、图像字幕制作及文本生成等技术训练机器翻译的完整代码示例](https://mp.weixin.qq.com/s/2gkdMwbc4lBCs3cIc9tfLw)
### Op教程
- [fc转conv](hthttps://github.com/BVLC/caffe/blob/master/examples/net_surgery.ipynb)
### 领域综述
- [11分钟训练Imagenet](https://mp.weixin.qq.com/s/KsVrYuv8hpwaB4uPTWEt_g?spm=5176.100239.blogcont231863.10.3715505e7AP9vx)
- [ MTL翻译地址](https://www.jiqizhixin.com/articles/2017-06-23-5)[原地址](http://ruder.io/multi-task/)
- [迁移学习](https://www.zhihu.com/question/41979241)
- [模型压缩](https://zhuanlan.zhihu.com/p/30548590)
- [语义分割](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#dilation)
- [优化算法](https://mp.weixin.qq.com/s/ABI1xtMTaFOqrDJ6MjlYzQ)
- [人类行为研究进展](https://mp.weixin.qq.com/s/4baaoCCdGX4iTw2MO_Y9rA)
- [GAN做图像翻译的一些总结](https://mp.weixin.qq.com/s/OZWih_xfqYdeP-x9MawogA)
### 横向对比
- [分布式框架](https://mp.weixin.qq.com/s/mc2UDI1QOYcbAShPdcN1WA)
- [优化GAN模型方法](https://mp.weixin.qq.com/s/xpnfhyiKgvgZWwarxHlbEg)
### 里程碑论文
#### 分割
- Mask R-CNN             论文提出一个概念上简单灵活通用的物体分割框架。
- [CapsNet](https://zhuanlan.zhihu.com/p/31262148) Hinton的大作胶囊网络
#### GAN
- [开山之作-Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661v1.pdf)
## 年度流行
- [2017python库](https://mp.weixin.qq.com/s/KWYET4jMTQwaydZmgCE4QA)
- [2017NIPS会议论文](https://www.zhihu.com/question/64943934)
## trick
- [加速预训练模型的方法](https://mp.weixin.qq.com/s/mLeTdwarWmz_CUcU1aerIw)
- [DNN训练trips-
魏秀参](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)