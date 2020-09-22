import numpy as np
import struct

from PIL import Image
import os

# 训练集数据
data_file = '../data/MNIST/raw/train-images-idx3-ubyte'
# It's 47040016B, but we should set to 47040000B
data_file_size = 47040016 # 怎么来的？ 28x28x60000， 因为数据集中的image size是28x28像素的
data_file_size = str(data_file_size - 16) + 'B' # 多少字节

data_buf = open(data_file, 'rb').read()

magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)

datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))

datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

# 加载训练集label
label_file = '../data/MNIST/raw/train-labels-idx1-ubyte'

# It's 60008B, but we should set to 60000B
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'train/'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)
# 这个for是用来进行十个数字分类0 - 9建立文件夹0 - 9十个文件夹
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

#
for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    file_name = datas_root + os.sep + str(label) + os.sep + \
                'mnist_train_' + str(ii) + '.png'
    img.save(file_name)