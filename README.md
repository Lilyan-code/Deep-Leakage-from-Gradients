# Deep Leakage from Gradients

这个代码是阅读DLG的paper和原作者的代码进行的复现。

因为是初学者， 此代码加入了自己的一些理解注释。

采用命令行调用代码执行。（Windows、Linux、Mac）兼容使用。

运行mnist数据集的时候因为画图的原因， 显示出来并不是灰度图像。需要在两个地方修改代码。

代码的222行处

```python
 plt.imshow(tp(gt_data[imidx].cpu()), cmap='gray') # 这一行是显示真实图片的意思
```



代码225行处

```python
plt.imshow(history[i][imidx], cmap='gray') # 在figure显示history存储假的图片数据
```



### 如何运行代码

在mnist上运行，需要按上述要求改写代码（才能出现灰度图像）

```shell
python ./DLG.py --index num（你想要还原的图片在数据集的下标） --dataset mnist or MNIST
```

不是灰度图像运行结果

![](https://cdn.jsdelivr.net/gh/Lyli724/Blogs_image@master/uPic/DLG_on_[0]_00000.png)

更改画图得到灰度图像

![](https://cdn.jsdelivr.net/gh/Lyli724/Blogs_image@master/uPic/DLG_on_[1000]_01000.png)

在cifar10上运行

```python
python ./DLG.py --index num (你想要还原的图片在数据集的下标) --dataset cifar10 or CIFAR10
```

运行结果

![](https://cdn.jsdelivr.net/gh/Lyli724/Blogs_image@master/uPic/DLG_on_[100]_00100.png)

在cifar100上运行

```python
python ./DLG.py --index num (你想要还原的图片在数据集的下标) --dataset cifar100 or CIFAR100
```

运行结果

![](https://cdn.jsdelivr.net/gh/Lyli724/Blogs_image@master/uPic/DLG_on_[1]_00001.png)

>  extracted_mnist文件夹：里面存储了解压mnist数据集的代码，运行即可解压mnist数据集
>
> extracted_cifar10文件夹：里面存储了解压cifar10数据集的代码，运行即可解压cifar10数据集
>
> extracted_cifar100文件夹：里面存储了解压cifar100数据集的代码，运行即可解压cifar100数据集
>
> 可以解压后查看数据集下标
>cifar10解压文件夹中需要创建一个空的test和train，以便于程序不会报错









