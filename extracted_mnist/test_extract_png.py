import numpy as np
import struct

from PIL import Image
import os

data_file = '../data/MNIST/raw/t10k-images-idx3-ubyte'
# It's 7840016B, but we should set to 7840000B
data_file_size = 7840016 # 28 x 28 x 10000
data_file_size = str(data_file_size - 16) + 'B'

data_buf = open(data_file, 'rb').read()

magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

label_file = '../data/MNIST/raw/t10k-labels-idx1-ubyte'

# It's 10008B, but we should set to 10000B
label_file_size = 10008
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'test/'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    # 这里就是生成文件label 就是0 - 9对应的也是文件夹编号，ii代表现在的一个图片是多少
    file_name = datas_root + os.sep + str(label) + os.sep + \
                'mnist_test_' + str(ii) + '.png'
    img.save(file_name)