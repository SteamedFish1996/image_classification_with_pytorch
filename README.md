# PyTorch Image Classification

## Installation
### win10 GPU
* [cuda](https://developer.nvidia.com/cuda-downloads) 
* [cudnn](https://developer.nvidia.com/cudnn)
* [可以参照这个教程安装](https://blog.csdn.net/qq_37296487/article/details/83028394)
* 离线安装包地址   59.77.7.59:/home/ps/zzy/

### pytorch
* [python3 with anaconda](https://www.anaconda.com/)
* [pytorch](https://pytorch.org/)（国内源坏了）离线安装包地址   59.77.7.59:/home/ps/zzy/win64
* 使用pip 或者conda 安装[requirement.txt](requirement.txt)


## About Pretrained Model

目前关于Pre-Training的最好的理解是，它可以让模型分配到一个很好的初始搜索空间，按照[Erhan09, Sec 4.2]中说法：The advantage of pre-training could be that it puts us in a region of parameter spacewhere basins of attraction run deeper than when picking starting parametersat random. The advantage would be due to a better optimization.来自Bengio的采访稿的一段，~Link~通常来讲，我所知道的模型都会受到不可计算性的影响（至少从理论上看，训练过程非常困难）。SVM之类的模型不会受到此类影响，但是如果你没有找到合适的特征空间，这些模型的普适性会受到影响。（寻找是非常困难的，深度学习正是解决了寻找特征空间的问题）。从Bengio的观点来看，Pre-Training带来的搜索空间，不仅有更好的计算（Optimation）性，还有更好的普适（Generalization）性。
