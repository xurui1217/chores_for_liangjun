from torch import nn
from torch.nn import functional as F
import torch


class ResidualBlock(nn.Module):
    '''
    Module: Residual Block
    '''

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        resisdual = x if self.right is None else self.right(x)
        out += resisdual
        return F.relu(out)


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
class discriminator(nn.Module):
    def __init__(self, input_channels=16):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    '''
    實現主Module: ResNet34
    ResNet34包含多個layer，每個layer又包含多個Residual block
    用子Module來實現Residual Block，用make_layer函數來實現layer
    '''

    def __init__(self):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        # 分類的Layer，分別有3, 4, 6個Residual Block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 分類用的Fully Connection
        # self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        構建Layer，包含多個Residual Block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        f1 = self.pre(x)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        # x = x.view(x.size(0), -1)
        return f1, f2, f3, f4, f5

# def load_pretrained_weight(model,path=None):
#     if path is not None:
#         state = torch.load(path)
#     else:
#         raise EOFError
#
#     model_dict = model.state_dict()
#     pretrained_dict = {k:v for k,v in state.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class Decoder34(nn.Module):

    def __init__(self, num_classes=9, backbone=None):
        super(Decoder34, self).__init__()
        self.backbone = backbone
        self.center = DecoderBlock(512, 1024, 512)
        self.de1 = DecoderBlock(1024, 512, 256)
        self.de2 = DecoderBlock(512, 256, 128)
        self.de3 = DecoderBlock(256, 128, 64)
        self.de4 = DecoderBlock(128, 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        for name, m in self.named_modules():
            if 'backbone' in name:
                continue
            else:
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f1, f2, f3, f4, f5 = self.backbone(x)
        center = self.center(f5)
        c4 = self.de1(torch.cat([center, f4], 1))
        c3 = self.de2(torch.cat([c4, f3], 1))
        c2 = self.de3(torch.cat([c3, f2], 1))
        c1 = self.de4(torch.cat([c2, f1], 1))
        x = self.final(c1)
        return x


if __name__ == '__main__':
    res = ResNet()
    # load_pretrained_weight(res,'../weight/resnet34-333f7ec4.pth')
    res.load_state_dict(torch.load(
        '../weight/resnet34-333f7ec4.pth'), strict=False)

    decoder = Decoder34(num_classes=9, backbone=res).cuda()
    # net = FCN8s(num_classes=9,pretrained=False).cuda()
    inp = torch.randn((4, 3, 480, 640))
    out = decoder(inp.cuda())
    print(out.shape)
