import torch.nn as nn
import numpy as np
import torch
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        return x



def main():
    # net = testNet()
    a=torch.rand(16,3,224,224)
    print(a.shape)
    conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
    a=conv1(a)
    a=a.reshape(-1,224*224)
    print(a.shape)

if __name__ == '__main__':
    main()