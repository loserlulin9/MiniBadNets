from torch import nn
import torch.nn.functional as F

"""
BadNet的结构仿照原文中的表1进行搭建：
        input       filter      stride      output      activation
conv1   1*28*28     16*1*5*5      1         16*24*24       ReLU
pool1   16*24*24    average,2*2   2         16*12*12        /
conv2   16*12*12    32*16*5*5     1         32*8*8         ReLU
pool2   32*8*8      average,2*2   2         32*4*4          /
fc1     32*4*4      /             /         512            ReLU
fc2     512         /             /         10            Softmax  
"""
class BadNet(nn.Module):
    def __init__(self, input_size=3, output=10):
        super().__init__()
        self.input_size = input_size
        self.output = output
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if input_size == 3:
            self.fc_features = 800
        else:
            self.fc_features = 512
        self.fc1 = nn.Linear(self.fc_features, 512)
        self.fc2 = nn.Linear(512, output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, self.fc_features) # 展平为1维向量，相当于Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x

