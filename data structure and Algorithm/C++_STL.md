- [STL（Standard Template Library）标准模板库](https://github.com/selfboot/CS_Offer/blob/master/C%2B%2B/STL.md)，是一个C++软件库，包括容器、算法、迭代器、函数、配接器、配制器六大部分。
![](https://camo.githubusercontent.com/cc158069ddd6be9d395b84b27e5c743f53c3636d/68747470733a2f2f63732d6f666665722d313235313733363636342e636f732e61702d6265696a696e672e6d7971636c6f75642e636f6d2f432b2b5f53544c5f312e706e67)
- 容器（containers）STL容器是将运用最广的一些数据结构,如 vector，list，deque，set，map等实现出来。  从实现的角度来看，容器是一种class_template。
- 算法（algorithms）提供了执行各种操作的方式，包括对容器内容执行初始化、排序、搜索和转换等 从实现的角度来看，STL算法是一种function_template。
- 迭代器（iterators）迭代器用于遍历对象集合的元素，扮演容器与算法之间的胶合剂，是所谓的泛型指针 从实现角度来看，迭代器是一种将operator*，operator->，operator++，operator--等指针操作予以重载的class_template。
- 仿函数（functors）行为类似函数，可作为算法的某种策略. 从实现角度来看，仿函数是一种重载了 operator()的class或者class_template。
- 配接器（Adaptor）一种用来修饰容器或者仿函数或迭代器接口的东西。
- 配制器（allocator）负责空间的配置与管理。 从实现的角度来看，配置器是一个实现了 动态配置空间、空间管理、空间释放的 class template。
- [STL笔试面试题总结（干货）](http://www.ishenping.com/ArtInfo/275544.html)

