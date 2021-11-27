import torch
from torch import nn


class CBAM_Module(nn.Module):

    def __init__(self, channels, reduction):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        # print(mx.shape)
        # print(avg.shape)
        # print(mx)
        # print(avg)
        x = avg + mx
        # print(x)
        x = self.sigmoid_channel(x)
        # print(x)
        # Spatial attention module
        x = module_input * x
        print(x.shape)
        print(x)
        module_input = x
        avg = torch.mean(x, 1, True)
        # print(avg.shape)
        # print(avg)
        mx, _ = torch.max(x, 1, True)
        # print(mx.shape)
        # print(mx)
        x = torch.cat((avg, mx), 1)
        # print(x.shape)
        # print(x)
        x = self.conv_after_concat(x)
        # print(x.shape)
        # print(x)
        x = self.sigmoid_spatial(x)
        # print(x)
        x = module_input * x
        return x


def mian():
    # pass
    x = torch.randn(1,16, 5, 5)
    model = CBAM_Module(16,4)
    out = model(x)



if __name__ == '__main__':
    mian()