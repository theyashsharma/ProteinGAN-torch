import torch
import torch.nn as nn
import torch.nn.functional as F
from common.model import ops
from common.model.ops import leaky_relu
from common.model.generator_ops import block

class generator_fully_connected(nn.Module):
    def __init__(self, input_dim, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
                              pooling='avg', scope_name='Generator', reuse=False):
        super().__init__()

        self.fun = nn.Sequential(
            ops.Linear(input_dim, gf_dim),
            ops.Linear(gf_dim, 512 * 21),
            nn.Tanh()
        )

    def forward(self, zs):
        output = self.fun(zs)
        output = torch.reshape(output, (-1, 1, 512, 21))
        return output

class original_generator(nn.Module):

    def __init__(self, zs_dim, labels, gf_dim, num_classes, kernel=3, stride=2, pooling='avg'):
        super().__init__()

        self.gf_dim = gf_dim

        self.linear = nn.Sequential(
            nn.Linear(zs_dim, 512 * 21 * gf_dim),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=(1, 1), mode='nearest'),
            nn.Conv2d(gf_dim, gf_dim, kernel_size = (3, 3), padding = 'same'),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=(1, 1), mode='nearest'),
            nn.Conv2d(gf_dim, gf_dim // 2, kernel_size = (3, 3), padding = 'same'),
            nn.ReLU(),
            nn.Conv2d(gf_dim // 2, 1, kernel_size = (3, 3), padding = 'same'),
            nn.Tanh()
        )

    def forward(self, zs):

        linear_out = self.linear(zs)

        linear_out = torch.reshape(linear_out, (-1, self.gf_dim, 512, 21))

        conv2_out = self.conv2(linear_out)
        conv3_out = self.conv3(conv2_out)
        return conv3_out

class generator_resnet(nn.Module):

    def __init__(self, zs_dim, labels, gf_dim, num_classes, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
            pooling='avg'):

        self.gf_dim  = gf_dim
        super().__init__()
        self.linear = ops.Linear(zs_dim, gf_dim * 512 * 21 * 8)

        self.block_stk = nn.Sequential(
            block(gf_dim * 8, gf_dim * 8),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 8, gf_dim * 4),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 4, gf_dim),
            nn.AvgPool2d((2, 2), stride=2),
            ops.BatchNorm(gf_dim)
        )
        self.conv_stk = nn.Sequential(
            nn.ReLU(),
            ops.conv2d(gf_dim, 1, kernel_size=(3, 3)),
            nn.Tanh()
        )

    def forward(self, zs):
        linear_out = self.linear(zs)
        linear_out = torch.reshape(linear_out, (-1, self.gf_dim * 8, 512, 21))

        block_out = self.block_stk(linear_out)
        conv_out = self.conv_stk(block_out)
        return conv_out
