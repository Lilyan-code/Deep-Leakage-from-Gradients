import time # 时间模块， 用于记录每次迭代的时间打印到控制台
import os # 操作系统模块， 用于创建文件夹，以及路径的一些操作
import numpy as np
import matplotlib.pyplot as plt
import torch # 导入pytorch核心模块
import torch.nn as nn # nn模块是pytorch和神经网络相关的模块
from torch.utils.data import Dataset # 从torch.utils模块获取数据集的操作的模块
from torchvision import datasets, transforms # 获取datasets和transforms模块，这两个模块主要用于处理数据。
# datasets是加载数据，transforms是将数据进行处理，
# 比如将PILImage图像转换成Tensor，或者将Tensor转换成PILImage图像
import pickle # pickle模块主要是用来将对象序列化存储到硬盘
import PIL.Image as Image # 处理图像数据的模块
import argparse # 参数命令行交互模块

os.environ['KMP_DUPLICATE_LIB_OK']='True'


'''
argparse实际上就是与命令行交互的一个模块。使用命令行选择一些args执行。
如: python DLG.py --index 0 选择index为0的下标
通过命令行选择可调节参数
argparse 模块可以让人轻松编写用户友好的命令行接口。
程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。 
argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
preference： https://docs.python.org/zh-cn/3/library/argparse.html
'''
parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.') # 使用 argparse 的第一步是创建一个 ArgumentParser 对象：
'''
给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
通常，这些调用指定 ArgumentParser 如何获取命令行字符串并将其转换为对象。这些信息在 parse_args() 调用时被存储和使用。
要想使用参考preference
'''
parser.add_argument('--index', type=int, default="10", # 这个参数是指明数据集下标，选择图片的下标
                    help='the index for leaking images on Dataset.')
parser.add_argument('--image', type=str, default="", # 这个参数是加载自己电脑的图片文件，输入在本地的图片地址即可
                    help='the path to customized image.')
parser.add_argument('--dataset', type=str, default="CIFAR10",
                    help='choose your dataset')
args = parser.parse_args()

'''
人脸数据集的一些处理
'''
class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs # img paths
        self.labs = labs # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab


def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)

    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst



'''
自定义LeNet网络
'''
class LeNet(nn.Module): # nn.Module, 定义神经网络必须继承的模块， 框架规定的形式
    def __init__(self, channel=3, hidden=768, num_classes=10): # 假设输入cifar10数据集， 默认3通道， 隐层维度为768， 分类为10
        super(LeNet, self).__init__() # 继承pytorch神经网络工具箱中的模块
        act = nn.Sigmoid # 激活函数为Sigmoid
        # nn.Sequential: 顺序容器。 模块将按照在构造函数中传递的顺序添加到模块中。 或者，也可以传递模块的有序字典
        self.body = nn.Sequential( # 设计神经网络结构，对于nn.Sequential.Preference : https://zhuanlan.zhihu.com/p/75206669
            # 设计输入通道为channel，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为2的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act(),
            # 设计输入通道为12，输出通道为12， 5x5卷积核尺寸，填充为5 // 2是整除。故填充为2， 步长为1的卷积层
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            # 经过卷积后， 使用Sigmoid激活函数激活
            act()
        )
        # 设计一个全连接映射层， 将hidden隐藏层映射到十个分类标签
        self.fc = nn.Sequential(
            nn.Linear(hidden, num_classes)
        )

    # 设计前向传播算法
    def forward(self, x):
        out = self.body(x) # 先经过nn.Sequential的顺序层得到一个输出
        out = out.view(out.size(0), -1) # 将输出转换对应的维度
        out = self.fc(out) # 最后将输出映射到一个十分类的一个列向量
        return out

'''
init weights
'''
def weights_init(m):
    try:
        if hasattr(m, "weight"): # hasattr：函数用于判断对象是否包含对应的属性。
            m.weight.data.uniform_(-0.5, 0.5) # 对m.weight.data进行均值初始化。m.weights.data指的是网络中的卷积核的权重
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"): # 对偏置进行初始化
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def main():
    seed = 1234 # 经过专家的实验， 随机种子数为1234结果会较好
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = args.dataset # 获得命令行输入的dataset
    root_path = '.'
    data_path = os.path.join(root_path, './data').replace('\\', '/')  # 指定数据存放的路径地址， replace是进行转义
    save_path = os.path.join(root_path, 'results/DLG_%s' % dataset).replace('\\', '/')  # 图片保存的路径

    lr = 1.0 # 学习率
    num_dummy = 1 # 一次输入还原的图片数量
    iteration = 300 # 一张图片迭代的次数
    num_exp = 1 # 实验次数也就是神经网络训练的epoch
    use_cuda = torch.cuda.is_available() # 是否可以使用gpu，返回值为True或False
    device = 'cuda' if use_cuda else 'cpu' # 设置 device是cpu或者cuda

    tt = transforms.Compose([transforms.ToTensor()]) # 将图像类型数据（PILImage）转换成Tensor张量
    tp = transforms.Compose([transforms.ToPILImage()]) # 将Tensor张量转换成图像类型数据

    '''
    打印路径而已
    '''
    print(dataset, 'root_path:', root_path)
    print(dataset, 'data_path:', data_path)
    print(dataset, 'save_path:', save_path)

    if not os.path.exists('results'): # 判断是否存在results文件夹，没有就创建，Linux中mkdir创建文件夹
        os.mkdir('results')
    if not os.path.exists(save_path): # 是否存在路径， 不存在则创建保存图片的路径
        os.mkdir(save_path)

    '''
    加载数据
    '''
    if dataset == 'MNIST' or dataset == 'mnist':  # 判断是什么数据集
        image_shape = (28, 28)  # mnist数据集图片尺寸是28x28
        num_classes = 10  # mnist数据分类为十分类： 0 ～ 9
        channel = 1  # mnist数据集是灰度图像所以是单通道
        hidden = 588  # hidden是神经网络最后一层全连接层的维度
        dst = datasets.MNIST(data_path, download=True)

    elif dataset == 'cifar10' or dataset == 'CIFAR10':
        image_shape = (32, 32)  # cifar10数据集图片尺寸是32x32
        num_classes = 10  # cifar10数据分类为十分类：卡车、 飞机等
        channel = 3  # cifar10数据集是RGB图像所以是三通道
        hidden = 768  # hidden是神经网络最后一层全连接层的维度
        dst = datasets.CIFAR10(data_path, download=True)

    elif dataset == 'cifar100' or dataset == 'CIFAR100':
        image_shape = (32, 32)  # cifar100数据集图片尺寸是32x32
        num_classes = 100  # cifar100数据分类为一百个分类
        channel = 3  # cifar100数据集是灰度图像所以是单通道
        hidden = 768  # hidden是神经网络最后一层全连接层的维度
        dst = datasets.CIFAR100(data_path, download=True)
    elif dataset == 'lfw':
        shape_img = (32, 32)
        num_classes = 5749
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, './data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)
    else:
        exit('unkown dataset')  # 未定义的数据集

    for idx_net in range(num_exp):
        net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes) # 初始化LeNet模型
        net.apply(weights_init) # 初始化模型中的卷积核的权重

        print('running %d|%d experiment' % (idx_net, num_exp))
        net = net.to(device)

        print('%s, Try to generate %d images' % ('DLG', num_dummy))

        criterion = nn.CrossEntropyLoss().to(device) # 设置损失函数为交叉熵函数
        imidx_list = [] # 用于记录当前还原图片的下标

        for imidx in range(num_dummy):
            idx = args.index # 从命令行获取还原图片的index
            imidx_list.append(idx) # 将index加入到列表中
            tmp_datum = tt(dst[idx][0]).float().to(device) # 将数据集中index对应的图片数据拿出来转换成Tensor张量
            tmp_datum = tmp_datum.view(1, *tmp_datum.size()) # 将tmp_datum数据重构形状， 可以用shape打印出来看看
            tmp_label = torch.Tensor([dst[idx][1]]).long().to(device) # 将数据集中index对应的图片的标签拿出来转换成Tensor张量
            tmp_label = tmp_label.view(1, ) # 将标签重塑为列向量形式
            if imidx == 0: # 如果imidx为0， 代表只处理一张图片
                gt_data = tmp_datum # gt_data表示真实图片数据
                gt_label = tmp_label # gt_label 表示真实图片的标签
            else:
                gt_data = torch.cat((gt_data, tmp_datum), dim=0) # 如果是多张图片就要将数据cat拼接起来
                gt_label = torch.cat((gt_label, tmp_label), dim=0)

            # compute original gradient
            out = net(gt_data) # 将真实图片数据丢入到net网络中获得一个预测的输出
            y = criterion(out, gt_label) # 使用交叉熵误差函数计算真实数据的预测输出和真实标签的误差
            dy_dx = torch.autograd.grad(y, net.parameters()) # 通过自动求微分得到真实梯度
            # 这一步是一个列表推导式，先从dy_dx这个Tensor中一步一步取元素出来，对原有的tensor进行克隆， 放在一个list中
            # https://blog.csdn.net/Answer3664/article/details/104417013
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))

            # generate dummy data and label。 生成假的数据和标签
            dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
            dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=lr) #设置优化器为拟牛顿法

        history = [] # 记录全部的假的数据（这里假的数据指的是随机产生的假图像）
        history_iters = [] # 记录画图使用的迭代次数
        grad_difference = [] # 记录真实梯度和虚假梯度的差
        data_difference = [] # 记录真实图片和虚假图片的差
        train_iters = [] #

        print('lr =', lr)
        for iters in range(iteration): # 开始训练迭代

            def closure(): # 闭包函数
                optimizer.zero_grad() # 每次都将梯度清零
                pred = net(dummy_data) # 将假的图片数据丢给神经网络求出预测的标签

                # 将假的预测进行softmax归一化，转换为概率
                dummy_loss = -torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
                    # dummy_loss = criterion(pred, gt_label)

                # 对假的数据进行自动微分， 求出假的梯度
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0 # 定义真实梯度和假梯度的差值
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): # 对应论文中的假的梯度减掉真的梯度平方的式子
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward() # 对||dw假 - dw真|| 进行反向传播进行更新
                return grad_diff

            optimizer.step(closure) # 优化器更新梯度
            current_loss = closure().item() # .item()方法是将Tensor中的元素转为值。item是得到一个元素张量里面的元素值
            train_iters.append(iters) # 将每次迭代次数append到列表中
            grad_difference.append(current_loss) # 将梯度差记录到losses列表中
            data_difference.append(torch.mean((dummy_data - gt_data) ** 2).item()) # 记录数据差

            if iters % int(iteration / 30) == 0: # 这一行是代表多少个iters画一张图
                current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) # 每次迭代打印出来时间
                print(current_time, iters, '梯度差 = %.8f, 数据差 = %.8f' % (current_loss, data_difference[-1])) # 打印出梯度差和数据差
                history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)]) # history 记录的是假的图片数据
                history_iters.append(iters) # 记录迭代次数用于画图使用

                for imidx in range(num_dummy): # 这个循环是迭代有多少张图片输入
                    plt.figure(figsize=(12, 8)) # plt.figure(figsize=())让图画在画布上， 并且使用figsize指定画布的大小（传入参数为元组）
                    plt.subplot(3, 10, 1) # 在figure画布上画子图的意思
                    # plt.imshow(tp(gt_data[imidx].cpu())) # 这一行是显示真实图片的意思, 如果是mnist数据集，将这一行改为如下
                    plt.imshow(tp(gt_data[imidx].cpu()), cmap='gray') # 灰度图像
                    for i in range(min(len(history), 29)): # 这一行是迭代画出子图的意思
                        plt.subplot(3, 10, i + 2)
                        # plt.imshow(history[i][imidx]) # 在figure显示history存储假的图片数据
                        plt.imshow(history[i][imidx], cmap='gray') # 显示灰度图像
                        plt.title('iter=%d' % (history_iters[i])) # 第几次迭代
                        plt.axis('off')

                    plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx])) # 保存图片地址
                    plt.close()

                if current_loss < 0.000001:  # converge
                    break

            loss_DLG = grad_difference # 梯度差
            label_DLG = torch.argmax(dummy_label, dim=-1).detach().item() # 求虚假数据产生的标签， 方便和真实图片产生的标签进行比较
            mse_DLG = data_difference # 数据差

    print('imidx_list 图片的index :', imidx_list)
    print('梯度差 :', loss_DLG[-1])
    print('数据差 :', mse_DLG[-1])
    print('gt_label 真实标签 :', gt_label.detach().cpu().data.numpy(), '虚假的还原数据标签: ', label_DLG)

    print('----------------------\n\n')


if __name__ == '__main__':
    main()



