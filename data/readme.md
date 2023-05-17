# 文件夹说明
|数据集|TrainSet|TestSet|说明|
|:----|:------:|:-----:|:--|
|DRealBSR|×|√|郭星贝制作的数据集，不清楚是否包含训练集和测试集的划分，具体可以问星贝|
|DRealSR|×|√|github上公布的DRealSR数据集，因没有训练集大图的数据，只能挑选出文字内容的测试图片|
|DRealSR_new|√|√|基于原始数据重新对齐制作的DRealSR新数据集,挑选了文字内容图片分别作为数据集和测试集|
|RealSR|√|√|RealSR数据集，从测试集和数据集挑选出的文字内容的图片|

<br>

**注意** 

- 所有图片均为大图，可能文字内容只占一小部分，有需要可以重新剪裁，剪裁代码为`code/crop.ipynb`
- DRealSR_new的训练集在使用时可能有切patch的需求（patch大小参照github公布的DRealSR数据集），切patch的代码为`code/get_patch.py`

<br>

# 文件夹结构
```目录结构
textSR/
├── DRealBSR/
│   └── test/
│   │   ├── x2/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   ├── x3/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   └── x4/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
├── DRealSR/
│   └── test/
│   │   ├── x2/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   ├── x3/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   └── x4/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
├── DRealSR_new/
│   ├── test/
│   │   ├── x2/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   ├── x3/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   └── x4/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   └── train/
│   │   ├── x2/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
│   │   ├── x3/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
│   │   └── x4/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
├── RealSR/
│   ├── test/
│   │   ├── x2/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   ├── x3/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   │   └── x4/
│   │   │   ├── test_HR/
│   │   │   └── test_LR/
│   └── train/
│   │   ├── x2/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
│   │   ├── x3/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
│   │   └── x4/
│   │   │   ├── train_HR/
│   │   │   └── train_LR/
├── code/
│   ├── crop.ipynb
│   └── get_patch.py
└── readme.md