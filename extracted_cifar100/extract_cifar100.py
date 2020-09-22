import cv2
import numpy as np
import pickle
import os


# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def cifar100_to_images():
    tar_dir = '../data/cifar-100-python/'  # 原始数据库目录
    train_root_dir = './train/'  # 图片保存目录
    test_root_dir = './test/'
    if not os.path.exists(train_root_dir):
        os.makedirs(train_root_dir)
    if not os.path.exists(test_root_dir):
        os.makedirs(test_root_dir)

    # 获取label对应的class，分为20个coarse class，共100个 fine class
    meta_Name = tar_dir + "meta"
    Meta_dic = unpickle(meta_Name)
    coarse_label_names = Meta_dic['coarse_label_names']
    fine_label_names = Meta_dic['fine_label_names']
    print(fine_label_names)

    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    dataName = tar_dir + "train"
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")
    for i in range(0, Xtr['data'].shape[0]):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        ###img_name:fine_label+coarse_label+fine_class+coarse_class+index
        picName = train_root_dir + str(Xtr['fine_labels'][i]) + '_' + str(Xtr['coarse_labels'][i]) + '_&' + \
                  fine_label_names[Xtr['fine_labels'][i]] + '&_' + coarse_label_names[
                      Xtr['coarse_labels'][i]] + '_' + str(i) + '.jpg'
        cv2.imwrite(picName, img)
    print(dataName + " loaded.")

    print("test_batch is loading...")
    # 生成测试集图片
    testXtr = unpickle(tar_dir + "test")
    for i in range(0, testXtr['data'].shape[0]):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        picName = test_root_dir + str(testXtr['fine_labels'][i]) + '_' + str(testXtr['coarse_labels'][i]) + '_&' + \
                  fine_label_names[testXtr['fine_labels'][i]] + '&_' + coarse_label_names[
                      testXtr['coarse_labels'][i]] + '_' + str(i) + '.jpg'
        cv2.imwrite(picName, img)
    print("test_batch loaded.")