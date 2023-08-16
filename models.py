import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#@torch.compile
class Lambda(nn.Module):
    # Custom layer
    def __init__(self, func: callable):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)
    
#@torch.compile
class ResidualBlock(nn.Module): 
    def __init__(self, in_features_dim):
        # in_features_dim 
        super(ResidualBlock, self).__init__()

        # My Residual Blocks
        self.conv_block = nn.Sequential(OrderedDict([
            ('tocuda', Lambda(lambda x: x.cuda())),
            ('reflect1', nn.ReflectionPad2d(1)),
            ('conv1', nn.Conv2d(in_features_dim, in_features_dim, 3)),
            ('ins_norm1', nn.InstanceNorm2d(in_features_dim)),
            ('relu', nn.ReLU(inplace=True)),
            ('reflect2', nn.ReflectionPad2d(1)),
            ('conv2', nn.Conv2d(in_features_dim, in_features_dim, 3)),
            ('ins_norm2', nn.InstanceNorm2d(in_features_dim))
        ]))

    def forward(self, x):
        return x + self.conv_block(x)
    
#@torch.compile
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [Lambda(lambda x: x.cuda()),]
        # Init the Residual Block      
        model += [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsample
        in_features_dim = 64 
        out_features_dim = in_features_dim*2 
        for i in range(2):
            model += [  nn.Conv2d(in_features_dim, out_features_dim, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features_dim),
                        nn.ReLU(inplace=True) ]
            in_features_dim = out_features_dim
            out_features_dim = in_features_dim*2

        # Residual Blocks
        for i in range(n_residual_blocks):
            model += [ResidualBlock(in_features_dim)]

        # Upsample
        out_features_dim = in_features_dim//2
        for i in range(2):
            model += [  nn.ConvTranspose2d(in_features_dim, out_features_dim, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features_dim),
                        nn.ReLU(inplace=True) ]
            in_features_dim = out_features_dim
            out_features_dim = in_features_dim//2

        # 输出层
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
#@torch.compile
class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [Lambda(lambda x: x.cuda()),]
        # Just Conv
        model += [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    # nn.BatchNorm2d(128) NoNoNoNo
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN
        model += [  nn.Conv2d(512, 1, 4, padding=1)  ]

        self.model = nn.Sequential(*model)

    def forward(self, x): 
        x = self.model(x)
        # torch.nn.AvgPool2d doesn't work...
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)