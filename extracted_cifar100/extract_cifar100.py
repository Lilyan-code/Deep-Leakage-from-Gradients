import os
import numpy as np
from scipy import misc

# 给我个路径我要创建文件呢
def makfile(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# 解压数据呢
def unpickle(file):
    import _pickle
    with open(file, 'rb') as fo:
        dict = _pickle.load(fo, encoding='bytes')
    return dict

# 给定路径添加数据呢
def Dealdata(adata, bdata, train=1, path='C:'):
    for aa, bb, c, d in zip(bdata[b'filenames'], bdata[b'fine_labels'], bdata[b'coarse_labels'], bdata[b'data']):
        print(adata[b'fine_label_names'][bb], adata[b'coarse_label_names'][c])
        if train:
            heng = os.path.join(path, 'train')
        else:
            heng = os.path.join(path, 'test')

        coarse_label_names = adata[b'coarse_label_names'][c]
        fine_label_names = adata[b'fine_label_names'][bb]
        # 路径已经准备好了
        path1 = os.path.join(heng, str(coarse_label_names, 'utf-8'))
        path2 = os.path.join(path1, str(fine_label_names, 'utf-8'))
        makfile(path2)
        path3 = os.path.join(path2, str(aa, 'utf-8'))

        # 再来一波data的数据呢
        picdata1 = np.reshape(d, (3, 32, 32))
        picdata = np.transpose(picdata1, (1, 2, 0))
        misc.imsave(path3, picdata)



# 解压后meta的路径
metapath = '../data/cifar-100-python/meta'
# 解压后test的路径
testpath = '../data/cifar-100-python/test'
# 解压后train的路径
trainpath = '../data/cifar-100-python/train'
# 存放图片的路径
storepicpath = '../data/cifar-100-python/'

a = unpickle(metapath)
b = unpickle(testpath)
c = unpickle(trainpath)
# 测试
Dealdata(a, b, train=0, path=storepicpath)
# 训练
Dealdata(a, c, train=1, path=storepicpath)




