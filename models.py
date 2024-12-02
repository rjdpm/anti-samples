#!/usr/bin/env python3

##############################################################################################
#% WRITER: RAJDEEP MONDAL            DATE: 20-11-2024
#% For bug and others mail me at: rdmondalofficial@gmail.com
#%--------------------------------------------------------------------------------------------
#% Code Citation: https://github.com/ayushkumartarun/zero-shot-unlearning/blob/main/models.py (Tarun et al., 2023)
#%                https://github.com/ayushkumartarun/deep-regression-unlearning/blob/main/models.py (Tarun et al., 2023)
##############################################################################################
import torch
from torch import nn
from torchinfo import summary
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18


__all__ = [
    'ResNet9_',
    'AllCNN_',
    'MobileNet',
    'ResNet18',
]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)
        

class AllCNN_(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True, padding = 0):
        super(AllCNN_, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        
        self.conv1 = Conv(n_channels, n_filter1, kernel_size=3, padding = padding, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout1 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout2 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm)
        if n_channels == 3:
            self.pool = nn.AvgPool2d(8)
        elif n_channels == 1:
            self.pool = nn.AvgPool2d(7)
        self.flatten = Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        actv1 = out
        
        out = self.conv2(out)
        actv2 = out
        
        out = self.conv3(out)
        actv3 = out
        
        out = self.dropout1(out)
        
        out = self.conv4(out)
        actv4 = out
        
        out = self.conv5(out)
        actv5 = out
        
        out = self.conv6(out)
        actv6 = out
        
        out = self.dropout2(out)
        
        out = self.conv7(out)
        actv7 = out
        
        out = self.conv8(out)
        actv8 = out
        
        out = self.pool(out)
        
        out = self.flatten(out)
        
        out = self.classifier(out)
        
        return out#, actv1, actv2, actv3, actv4, actv5, actv6, actv7, actv8 


class ResidualBlock_(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock_, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out = out + residual
        
        return out
    
    
    
class ResNet9_(nn.Module):
    """
    A Residual network.
    """
    def __init__(self, n_classes = 10, num_input_channels = 3, padding = 0):
        super(ResNet9_, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_input_channels, out_channels=64, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock_(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock_(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=256, out_features=n_classes, bias=True)

    def forward(self, x):
        for idx, layer in enumerate(self.conv):
            x = layer(x)
            if idx == 0:
                activation1 = x
            if idx == 3:
                activation2 = x
            if idx == 8:
                activation3 = x
            if idx == 12:
                activation4 = x
        
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)

        return x#, activation1, activation2, activation3, activation4    
    



class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large()
        base_list = [*list(base.children())[:-1]]
        self.conv_norm1 = nn.Sequential(*base_list[0][0])
        for i in range(1, 16):
            exec(f"self.inverted_residual_{i} = base_list[0][{i}]")
        self.conv_norm2 = nn.Sequential(*base_list[0][16])
        self.pool1 = base_list[1]
        self.drop = nn.Dropout()
        self.final = nn.Linear(960,1)
    
    def forward(self,x):
        actvn1 = self.conv_norm1(x)
        
        for i in range(1, 16):
            exec(f"actvn{i+1} = self.inverted_residual_{i}(actvn{i})", locals(), globals())
        
        actvn17 = self.conv_norm2(actvn16)
        out = self.pool1(actvn17)
        
        out = self.drop(out.view(-1,self.final.in_features))
        return self.final(out), actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7,\
                actvn8, actvn9, actvn10, actvn11, actvn12, actvn13, actvn14, actvn15,\
                actvn16, actvn17

    
class ResNet18(nn.Module):
    def __init__(self, n_classes = 10, num_input_channels = 3):
        super().__init__()
        base = resnet18(pretrained=False)
        in_features = base.fc.in_features
        base_list = [*list(base.children())[:-1]]
        self.layer1 = nn.Sequential(*base_list[0:3])
        self.pool1 = base_list[3]
        self.basic_block1 = base_list[4][0]
        self.basic_block2 = base_list[4][1]
        self.basic_block3 = base_list[5][0]
        self.basic_block4 = base_list[5][1]
        self.basic_block5 = base_list[6][0]
        self.basic_block6 = base_list[6][1]
        self.basic_block7 = base_list[7][0]
        self.basic_block8 = base_list[7][1]
        self.pool2 = base_list[8]
        self.drop = nn.Dropout()
        self.final = nn.Linear(512,n_classes)
        
    
    def forward(self,x):
        out = self.layer1(x)
        actvn1 = out
        
        out = self.pool1(out)
        
        out = self.basic_block1(out)
        actvn2 = out
        
        out = self.basic_block2(out)
        actvn3 = out
        
        out = self.basic_block3(out)
        actvn4 = out
        
        out = self.basic_block4(out)
        actvn5 = out
        
        out = self.basic_block5(out)
        actvn6 = out
        
        out = self.basic_block6(out)
        actvn7 = out
        
        out = self.basic_block7(out)
        actvn8 = out
        
        out = self.basic_block8(out)
        actvn9 = out
        
        out = self.pool2(out)
        out = out.view(-1,self.final.in_features)
            
        out = self.final(out)
        
        return out#, actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7, actvn8, actvn9 

    
    
# test network
if __name__ == '__main__':
    
    num_chanel = 3
    H, W = 32, 32#28, 28
    num_class = 10#3
    batch_size = 128
    model = AllCNN_(n_channels=num_chanel, num_classes=num_class, padding = 0)
    
    print('-'*70)
    print('Network architechture (num chanel-{}, num class- {}) as follows:' .format(num_chanel, num_class))
    print('-'*70)
    print(model)
    print('-'*70)
    
    print('Network summary:')
    print('-'*70)
    summary(model, (batch_size, num_chanel, H, W ), device=str("cpu"))
    print('-'*70)
    
    print('Network input output dims check')
    print('-'*70)
    x = torch.randn(batch_size, num_chanel, H, W)
    print(x.shape)
    y = model(x)
    print(y[0].shape)
    
    
