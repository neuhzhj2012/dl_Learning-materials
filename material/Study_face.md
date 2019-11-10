## 人脸研究方向 ##
### 综述文章
- [人脸脸型分类研究现状](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029461&idx=1&sn=f17970272a31e9eb4f00408338a3b382&chksm=87134528b064cc3e12a4ed75a492c5917dc2a2314f2e5ec13448d32dcc0eeeda9a4f20055f50&scene=21#wechat_redirect)
- [人脸颜值研究综述](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029259&idx=1&sn=4a981973b57fd1ba77049163bd886683&chksm=871344f6b064cde0003d960ad49ea27d89497e38fd4af555ab9e2c9b6c72c46f186c770e9edf&scene=21#wechat_redirect)
- [人脸年龄估计研究现状](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029213&idx=1&sn=e6f0cf21d8d4c3620d01bf80f7d536b5&chksm=87134420b064cd36ada3f3a524386b825ea62070e7c679cf56d4446189bda131e30d3febc541&scene=21#wechat_redirect)
- [人脸表情识别研究](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029493&idx=1&sn=3a6442bfbc2f1a917420adc3eed91272&chksm=87134508b064cc1ed0e23cee897946f7a7bd4dbd43e8516cc74a6760431229da7909006bf40a&scene=21#wechat_redirect)
- [如何降低遮挡对人脸识别的影响](https://mp.weixin.qq.com/s?__biz=MzA3NDIyMjM1NA==&mid=2649029586&idx=1&sn=0656db9c8fbc46cd7730618b44cae4d6&chksm=871345afb064ccb9da37071f47d1ed109cbfcb6b476cce93dd220c6bcd1c7d287150357e6b2d&scene=21#wechat_redirect)
### [人脸检测](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)  ###


- 定位图像中人脸的位置区域
- 常和人脸校正一起使用，获得角度校正后的人脸


### [人脸校正](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/) ###


- 定位人脸五官关键点
- 由眼睛相对位置计算人脸倾斜角度，根据设定的人脸大小和眼睛位置可得到基于眼睛中心的仿射变换矩阵，进而对人脸进行校正

### 人脸比对 ###
通过特征相似度比较人脸的算法，涉及到人脸特征、人脸验证、人脸检索和人脸识别等相关技术

##### 人脸特征 #####
- 将人脸由图像变为向量表示的过程

##### 人脸验证 #####
- 对比两张人脸图像，判断是否为同一张人脸


##### [人脸识别](https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/)  #####
- 得到图像中人脸对应的人名信息

##### 人脸检索 #####
- 待定人脸与数据库中的人脸进行相似度比较

### [人脸活体](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/) ###
- 判断人脸区域是否为现实中的人，而非照片，画像等

### 人脸属性 ###
- 根据人脸判断相关属性，如性别、年龄、表情等