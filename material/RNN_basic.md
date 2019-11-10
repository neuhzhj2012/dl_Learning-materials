### 相关概念
- 词频

1. 概念：词出现的频率
2. [分类](https://www.jianshu.com/p/6be7d4d85477):绝对词频和相对词频。前者表示词项在(当前/所有)文档中出现的频率。后者表示归一化的词频，用TF-IDF(Term Frequency-Inverse Document Frequency))表示，TF值越大，表示这个词项越重要；IDF值越大 表示该词项出现的文档数越少，它越能够把文档区分出来，于是就越重要。

![](https://math.jianshu.com/math?formula=TF%3D%5Cfrac%7B%E8%AF%A5%E8%AF%8D%E9%A1%B9%EF%BC%88Term%EF%BC%89%E5%9C%A8%E8%AF%A5%E6%96%87%E6%A1%A3%E5%87%BA%E7%8E%B0%E7%9A%84%E6%AC%A1%E6%95%B0%7D%7B%E8%AF%A5%E6%96%87%E6%A1%A3%E7%9A%84%E8%AF%8D%E9%A1%B9%E7%9A%84%E6%80%BB%E6%95%B0%7D)
![](https://math.jianshu.com/math?formula=IDF%3Dlog%5Cfrac%7B%E6%96%87%E6%A1%A3%E5%BA%93%E4%B8%AD%E7%9A%84%E6%96%87%E6%A1%A3%E6%80%BB%E6%95%B0%7D%7B%E5%8C%85%E5%90%AB%E8%AF%A5%E8%AF%8D%E9%A1%B9%E7%9A%84%E6%96%87%E6%A1%A3%E6%95%B0%20%2B%201%7D)