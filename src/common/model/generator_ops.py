import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ops


class _block(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, Conv=ops.conv2d, kernel=(3, 3), stride = (2, 2), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same', batch_norm=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm

        self.num_classes = num_classes
        if num_classes is not None:
            self.bn0 = ops.ConditionalBatchNorm(out_channels, num_classes)
            self.bn1 = ops.ConditionalBatchNorm(out_channels, num_classes)
        elif self.batch_norm:
            self.bn0 = ops.BatchNorm(out_channels)
            self.bn1 = ops.BatchNorm(out_channels)

        self.conv = Conv(in_channels, out_channels, kernel_size=kernel, dilation = dilation, padding=padding)
        self.act = Act()
        self.pooling = pooling
        self.stride = stride

        if self.pooling=='avg':
            self.upsample = torch.nn.Upsample(scale_factor=stride, mode='nearest')
            self.down_conv = Conv(out_channels, out_channels, kernel_size=kernel, padding=padding)
        elif self.pooling == 'conv':
            
            if Conv == ops.snconv2d:
                self.deconv = ops.sndeconv2d(out_channels, out_channels, kernel_size=kernel, stride=stride)
            elif Conv == ops.scaled_conv2d:
                self.deconv = ops.scaled_deconv2d(out_channels, out_channels, kernel_size=kernel, stride=stride)
            else:
                self.deconv = ops.deconv2d(out_channels, out_channels, kernel_size=kernel, stride=stride)
        elif pooling == 'subpixel':
            raise NotImplementedError
        elif pooling == 'None':
            self.down_conv = Conv(out_channels, out_channels, kernel_size=(1, 1), padding=padding)

        if self.pooling=='avg':
            self.upsample_0 = torch.nn.Upsample(scale_factor=stride, mode='nearest')
            self.down_conv_0 = Conv(in_channels, out_channels, kernel_size=kernel, padding=padding)
        elif self.pooling == 'conv':
            
            if Conv == ops.snconv2d:
                self.deconv_0 = ops.sndeconv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
            elif Conv == ops.scaled_conv2d:
                self.deconv_0 = ops.scaled_deconv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
            else:
                self.deconv_0 = ops.deconv2d(in_channels, out_channels, kernel_size=kernel, stride=stride)
        elif pooling == 'subpixel':
            raise NotImplementedError
        elif pooling == 'None':
            self.down_conv_0 = Conv(in_channels, out_channels, kernel_size=(1, 1), padding=padding)

    def forward(self, x, labels):
        x_0 = x
        x = self.conv(x)

        if self.num_classes is not None:
            x = self.bn0(x, labels)
        elif self.batch_norm:
            x = self.bn0(x)

        x = self.act(x)

        if self.pooling=='avg':
            x = self.upsample(x)
            x = self.down_conv(x)
        elif self.pooling == 'conv':
            x = self.deconv(x)
        elif self.pooling == 'None':
            x = self.down_conv(x)

        if self.num_classes is not None:
            x = self.bn1(x, labels)
        elif self.batch_norm:
            x = self.bn1(x)

        if self.pooling=='avg':
            x_0 = self.upsample_0(x_0)
            x_0 = self.down_conv_0(x_0)
        elif self.pooling == 'conv':
            x_0 = self.deconv_0(x_0)
        elif self.pooling == 'None':
            x_0 = self.down_conv_0(x_0)

        out = x_0 + x
        return out

class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride = (2, 2), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in GAN. It used standard 2D conv.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = ops.leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(None, in_channels, out_channels, ops.conv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x, None)

class block_conditional(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, kernel=(3, 3), stride = (1, 1), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in GAN. It used standard 2D conv.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = ops.leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(num_classes, in_channels, out_channels, ops.conv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x, labels):
        return self.blk(x, labels)


class sn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in SNGAN. It used 2D conv with spectral normalization.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = ops.leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(None, in_channels, out_channels, ops.snconv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x, None)


class sn_block_conditional(nn.Module):

    def __init__(self, num_classes, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in SNGAN. It used 2D conv with spectral normalization.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = ops.leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(num_classes, in_channels, out_channels, ops.snconv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x, labels):
        return self.blk(x, labels)

class scaled_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), strides=(1, 1), dilation=(1, 1), Act=ops.leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in ProGAN. It used 2D conv with Equalized learning rate.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections. (Default value = None)
          act: The activation function used in the block. (Default value = ops.leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(None, in_channels, out_channels, ops.scaled_conv2d, kernel, strides, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x, None)

class get_kernel(nn.Module):
    """
    Calculates the kernel size given the input. Kernel size is changed only if the input dimentions are smaller
    than kernel
    Args:
      x: The input vector.
      kernel:  The height and width of the convolution kernel filter
    Returns:
      The height and width of new convolution kernel filter
    """
    def __init__(self, x, kernel):
        height = self.kernel[0]
        width = self.kernel[1]

    def forward(self, x, height, width):
        if x.Size()[1] < height:
            height = x.Size()[1]
        elif x.Size()[2] < width:
            width = x.Size()[2]

        return (height, width)
