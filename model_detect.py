import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.LeakyReLU(0.1), bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if act is not None:
            layers.append(act)
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class Residual(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBnAct(ch, ch//2, k=1, p=0)
        self.conv2 = ConvBnAct(ch//2, ch, k=3, p=1)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class BalancedDetector(nn.Module):
    def __init__(self, S=13, B=1, C=0, dropout_p=0.3): 
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.output_dim = B * 5 + C

        self.stem = nn.Sequential(
            ConvBnAct(3, 32, k=3, p=1),
            nn.MaxPool2d(2,2)  
        )
        self.stage1 = nn.Sequential(
            ConvBnAct(32, 64, k=3, p=1),
            Residual(64),
            nn.MaxPool2d(2,2)  
        )
        self.stage2 = nn.Sequential(
            ConvBnAct(64, 128, k=3, p=1),
            Residual(128),
            nn.MaxPool2d(2,2) 
        )
        self.stage3 = nn.Sequential(
            ConvBnAct(128, 256, k=3, p=1),
            Residual(256),
            nn.MaxPool2d(2,2) 
        )
        self.stage4 = nn.Sequential(
            ConvBnAct(256, 256, k=3, p=1),
            nn.MaxPool2d(2,2)  
        )

        self.head = nn.Sequential(
            ConvBnAct(256, 128, k=1, p=0),
            ConvBnAct(128, 256, k=3, p=1),
            nn.Dropout2d(p=dropout_p, inplace=True),
            nn.Conv2d(256, self.output_dim, kernel_size=1, stride=1, padding=0)
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        x = x.permute(0,2,3,1)  
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)