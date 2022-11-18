import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# single: 对应到目标label
# all：label[i]-->label[i+1]

class PoisonedDataset(Dataset):
    # 添加两种类型的后门，返回带有中毒标签的图像和新的张量
    def __init__(self, dataset, trigger_label, proportion=0.1, mode="train", datasetname="mnist", attack="single"):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.datasetname = datasetname
        if attack == "single":
            self.data, self.targets = self.add_trigger(dataset.data, dataset.targets, trigger_label, proportion, mode)
        elif attack == "all":
            self.data, self.targets = self.add_trigger2(dataset.data, dataset.targets, proportion, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1
        label = torch.Tensor(label)
        return img, label

    # single--在图片的右下角加入三个white-pixels
    def add_trigger(self, data, targets, trigger_label, proportion, mode):
        print("## generate " + mode + " Bad Imgs")
        
        # np.copy()可以返回给定数组的深拷贝，只拷贝值，不拷贝地址，原数组变化不影响拷贝后的数组
        new_data = np.copy(data)
        new_targets = np.copy(targets)

        # 返回一个长度为proportion*length的随机序列，其中是0~len(data)大小的随机序列
        trig_list = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]

        if len(new_data.shape) == 3:  #Check whether there is the singleton dimension missing abd add it in the array, ie. for mnist 28x28x1 and for cifar 32x32x1
            new_data = np.expand_dims(new_data, axis=3)

        width, height, channels = new_data.shape[1:] # 深度学习中图像的四维应该是：batch, height, width, channel

        for i in trig_list:
            new_targets[i] = trigger_label # 改变加了trigger后图像的label
            for c in range(channels):
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255
        new_data = reshape_before_training(new_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(trig_list), len(new_data)-len(trig_list), proportion))
        # return Tensor
        return torch.Tensor(new_data), new_targets

    # All to all trigger
    def add_trigger2(self, data, targets, proportion, mode):
        print("## generate " + mode + " Bad Imgs")
        new_data = np.copy(data)
        new_targets = np.copy(targets)

        trig_list = np.random.permutation(len(new_data))[0: int(len(new_data) * proportion)]
        if len(new_data.shape) == 3:  
            new_data = np.expand_dims(new_data, axis=3)
        width, height, channels = new_data.shape[1:]
       
        for i in trig_list:
            if targets[i] == 9:
                new_targets[i] = 0
            else:
                new_targets[i] = targets[i] + 1
            for c in range(channels):
                new_data[i, width-3, height-3, c] = 255
                new_data[i, width-4, height-2, c] = 255
                new_data[i, width-2, height-4, c] = 255
                new_data[i, width-2, height-2, c] = 255

        new_data = reshape_before_training(new_data)
        print("Injecting Over: %d Bad Imgs, %d Clean Imgs (%.2f)" % (len(trig_list), len(new_data)-len(trig_list), proportion))
        # return Tensor
        return torch.Tensor(new_data), new_targets


# train之前的数据变换
def reshape_before_training(data):
    return np.array(data.reshape(len(data), data.shape[3], data.shape[2], data.shape[1])) # 改变三个维度表示的意义，变为channel, height, width


def vis_img(array):
    plt.imshow(array)
    plt.show()


