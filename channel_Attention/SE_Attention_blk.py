

import  torch
import  torch.nn as nn




class channel_attention_blk(nn.Module):
    def __init__(self, channel, reduction=16):  ############## 论文中实验表明 reduction=8或者16时效果最好，论文推荐reduction=16
        super(channel_attention_blk, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(x)
        y = self.avg_pool(x).view(b, c)
        # print(y)
        y = self.fc(y).view(b, c, 1, 1)
        print(y.shape)
        # print(y)
        print(y.expand_as(x).shape)
        # return x * y.expand_as(x)
        # return x * y



def main():
        x= torch.randn(1, 64, 224, 224)
        # print(x.shape)
        # q = channel_attention_blk(64)
        model = channel_attention_blk(64)
        outputs = model(x)
        # print(q)





if __name__ == '__main__':
    main()












