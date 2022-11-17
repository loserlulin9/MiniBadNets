from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PoisonedDataset
import sys

# In this function the two datasets are loaded. Either mnist or cifar10
# return:(train_data,test_data)--两个经过归一化处理的datasets.CIFAR10类，用于训练和测试

def load_sets(datasetname, download, dataset_path):
    try:
        if datasetname == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])
            train_data = datasets.MNIST(root=dataset_path, train=True, download=download, transform=transform)
            test_data = datasets.MNIST(root=dataset_path, train=False, download=download, transform=transform)
            return train_data, test_data

        elif datasetname == 'cifar':
            transform = transforms.Compose( # 将以下的操作进行组合
                [transforms.ToTensor(), # 转化成tensor格式，进入神经网络，这个时候像素数据已经被除以了225，所以进入normalize的数据就是0~1的数据
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) # 对像素值进行归一化处理，数据（在这里做了适当的修改）从ImageNet计算得出，如果需要使用自己的数据集，则可以自行计算均值、方差
            
            train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download, transform=transform)
            # root:数据集所在目录的根目录，在download设置为true的情况下，会保存在cifar-10-batches-py的目录下
            # train:为True的话则从训练集中创建数据集，否则从测试集中创建数据集
            # download:若为True则从Internet下载数据集，将其放在根目录中。如果已经下载，则不会再次下载
            # transform:一个函数变换，并返回转换后的版本

            test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download, transform=transform)
            return train_data, test_data
        else:
            raise NotAcceptedDataset

    except NotAcceptedDataset:
        print('Dataset Error. Choose "cifar" or "mnist"')
        sys.exit()


# With this function 3 dataloaders are returned. The first is the training dataloader with a portion of poisoned data,
# second is the test dataloader without any poisoned data to test the performance of the trained model, and the third is
# the dataloader with poisoned test data to test the poisoned model of new poisoned test data.


def backdoor_data_loader(datasetname, train_data, test_data, trigger_label, proportion, batch_size, attack):
    train_data = PoisonedDataset(train_data, trigger_label, proportion=proportion, mode="train", datasetname=datasetname, attack=attack) # 模型训练，投毒比为proportion
    test_data_orig = PoisonedDataset(test_data,  trigger_label, proportion=0, mode="test", datasetname=datasetname, attack=attack) # 原始图片
    test_data_trig = PoisonedDataset(test_data,  trigger_label, proportion=1, mode="test", datasetname=datasetname, attack=attack) # 返回的是全部加了trigger的有毒数据

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=False)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_orig_loader, test_data_trig_loader


# Just a simple custom exception that is raised when the dataset argument is not accepted


class NotAcceptedDataset(Exception):
    """Not accepted dataset as argument"""
    pass
