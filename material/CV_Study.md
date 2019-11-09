#### 多标签学习
- 研究内容
1. 综述文章[2014年A review on multi-label classification learning algorithms](https://www.cnblogs.com/liaohuiqiang/p/9339996.html)
2. 实现方法：多标签网络的本质是使用Sigmoid activation function替代Softmax激活函数
- 相关数据集
1. 数据集。[南京大学包含5类的共2k张图](http://lamda.nju.edu.cn/data_MIMLimage.ashx)
2. [如何构建自己的数据集](https://datascience.stackexchange.com/questions/26885/multilabel-image-classification-is-it-necessary-to-have-traning-data-for-each-c)
- [探讨如何在多标签分类问题中考虑标签间的相关性](https://zhuanlan.zhihu.com/p/39535198)
- 相关项目

1. [基于keras对南京大学数据集的分类教程](https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24)
2. [基于keras的衣服类型和颜色的多分类](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)

```
A，仅含猫的，仅含狗的，猫狗都有的作为样本集;
B，仅含猫的，仅含狗的;
上边两种情况只是正样本的选择方法，负样本的选择为不包含猫狗的图 假如无负样本，则网络会将相似数据分类为猫或狗
```
#### 图像模糊算法研究
1. 出发点。图像模糊研究属于IQA的一部分，也是衡量用户体验(Quality of Experience)的一个关键因素。

```
IQA（Image Quality Assessment）图像质量评价
```
2. [分类](http://imgtec.eetrend.com/d6-imgtec/blog/2018-07/16880.html)：(根据区域划分)全局和局部模糊。(根据模糊类型)衍射模糊、高斯模糊、散焦模糊、运动模糊。

```
高斯衍射模糊产生的主要原因是由于天气中的水雾或灰尘颗粒所引起的模糊，体现在所有的图像画面的朦胧或雨雾状的效果;
高斯模糊则是看起来有些像透过半透明玻璃看图像的效果，体现在所有的图像画面的朦胧或雨雾状的效果;
散焦模糊是因为物体没有在镜头的对焦的清晰范围内而产生的，体现在散焦部位的扩散效果;
运动模糊是由于物体快速运动或有抖动的情况下引起的模糊，表现为局部图像的拖尾效果
```
3. [图片清晰度](https://juejin.im/post/5b76df76f265da43330c3f50)Pech-Pacheco在2000年模式识别国际会议提出将图片中某一通道（一般用灰度值）通过拉普拉斯掩模做卷积运算，然后计算标准差，出来的值就可以代表图片清晰度。
4. 研究方向：基于特征的模糊判断和基于神经网络的判断。其中特征主要包括图像边缘、像素值的统计信息和变换域等；后者主要利用神经网络训练模糊分类数据集进行判定。

```
基于图像边缘特征判断的依据是模糊会导致边缘不明显；
基于像素值统计信息的依据是清晰图像的像素分布是规范的(如广义高斯分布)，模糊会改变该分布。对像素值统计分布参数的计算反应模糊程度；
基于变换域研究的依据是纹理特征对应傅里叶变换后的高频分量，模糊会减少纹理特征。变换后高频信息的丰富程度反映了模糊程度；
```
5. 解决方案：滤波去噪法、锐化法、改变图像信噪比、帧与帧的平均值。

```
滤波去噪法包括频域方法、空间域方法、矢量滤波。其中频域方法是采用低通波段法(如：维纳滤波)，以达到去掉图像中的噪声的目的。空间域方法选取局部领域中的中间像素值，然后使用高通波段用以增强边缘的高频信号。矢量滤波则是专门使用在因运动造成的图像模糊上。矢量滤波器的主要原理则是可以通过对模糊图像中的因移动所形成的平行四边形的模糊拖尾图形， 来估算出运动模式和运动速度，进而通过采用数学算法，估算出物体旋转、平移的量，从而实现恢复图像的清晰度；
锐化的实质是增强原图像的高频分量；
信噪比指的是图像的信号内容与所感知到的噪声的比值。调节信噪比可以使得信号的噪声及图像的模糊程度达到一个均衡的状态；
帧间的平均值是在所得到的多帧图像中，选定其中一帧较为清晰的图像，并对其进行叠加，用来求得帧的叠加效果或多帧平均值的效果。然后在这些操作的基础上继续使用相应的滤波器，从而实现在图像的固定对象中去除动态噪声的方法。

```
- [模糊度评价修正](https://cloud.tencent.com/developer/article/1051339)

1. 修正原因：模糊度评价和其他图像损伤如块损伤评价（马赛克效应）不一样的地方在于： 物理上同样程度的模糊，人眼对于其主观感受却不同(主观感受与图像的内容是强相关的)
2. 目的：得到更加真实的iqa结果
3. 解决方案：内容无关和内容检索的修正方法。前者是利用Logistic函数(其参数是IQA数据库上拟合得到)。该方法缺点是不同数据库上拟合的参数不同，导致结果不同，对中轻度模糊基本没有改善;后者利用主客观分值拟合后斜率聚类的结果获得每一类的校正参数，可提高轻度到中度损伤程度的评价准确性
4. [无参考图像清晰度评价方法](http://nkwavelet.blog.163.com/blog/static/227756038201461532247117)
- 相关项目

1. [Blur detection with OpenCV](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
2. [Mobile Image Blur Detection with Machine Learning](https://medium.com/snapaddy-tech-blog/mobile-image-blur-detection-with-machine-learning-c0b703eab7de)