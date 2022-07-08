import torch
import torch.nn as nn
import torch.nn.functional as F
from common.model import ops
from common.model.ops import block, leaky_relu


class discriminator_fully_connected(nn.Module):
    def __init__(self, in_shape, labels, df_dim, number_classes, kernel=3, stride=2, pooling='avg', update_collection=None):
        super().__init__()

        self.out = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_shape, df_dim),
            leaky_relu(),
            nn.Linear(df_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.out(x)   
        return output

class original_discriminator(nn.Module):
    def __init__(self, in_shape, labels, df_dim, number_classes, width, height, kernel=3, stride=1, pooling='avg', update_collection=None):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_shape, df_dim // 4, kernel_size = kernel, stride = stride, padding = 'same'),
            leaky_relu(),
            nn.Conv2d(df_dim // 4, df_dim // 2, kernel_size = kernel, stride = stride, padding = 'same'),
            leaky_relu(),
            nn.Conv2d(df_dim // 2, df_dim, kernel_size = kernel, stride = stride, padding = 'same'),
            leaky_relu()
        )
        self.linear = nn.Linear(width * height * df_dim, 1) #df_dim is wrong here
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

class discriminator_resnet(nn.Module):
    def __init__(self, in_shape, labels, df_dim, number_classes, width, height, kernel=(3, 3), strides=(2, 2), dilations=(1, 1),
            pooling='avg', act=leaky_relu):
        super().__init__()
        self.h0 = block(in_shape, df_dim, Act=act) # 12 * 12
        self.h1 = block(df_dim, df_dim * 2, Act=act) # 6 * 6
        self.h2 = block(df_dim * 2, df_dim * 4, Act=act) # 3 * 3
        self.h5 = block(df_dim * 4, df_dim * 8, Act=act) # 3 * 3
        self.h5_act = act()
        self.final = ops.Linear(df_dim * 8 * width * height, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)
        x = self.h0(x)
        x = self.h1(x)
        x = self.h2(x)
        x = self.h5(x)
        x = self.h5_act(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.final(x)
        x = self.sigmoid(x)
        return x
