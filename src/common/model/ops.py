import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO : cuda
#TODO : match exact output for deterministic funcs
#TODO : do a channel first - channel last test

class leaky_relu(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.m = nn.LeakyReLU(negative_slope=alpha)

    def forward(self, x):
        return self.m(x)

class ConditionalBatchNorm(nn.Module):
    """ Conditional BatchNorm.
        https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
    """

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

class BatchNorm(nn.Module):

    """ Input: NCHW"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum)

    def forward(self, x):
        return self.bn(x)

class Self_Attn(nn.Module):
    
    def __init__(self, in_channels, out_channels, sn=True):
        """ 
            Self-Attention Layer
        Args:
            in_channels : number of input channels
            out_channels : number of out channels
            sn : Flag for spectral normalization
        """
        super().__init__()
        self.out_channels = out_channels
        self.out_channels_sqrt = int(math.sqrt(out_channels))

        self.softmax = nn.Softmax(dim=-1)
        self.register_parameter(name='attention_multiplier', param=torch.nn.Parameter(torch.tensor(0.0)))

        if sn:
            self.f = snconv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
            self.g = snconv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
            self.h = snconv2d(in_channels, self.out_channels, kernel_size=(1, 1))
        else:
            self.f = conv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
            self.g = conv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
            self.h = conv2d(in_channels, self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        """ x -> NCHW """

        # TODO : seems like out_channels == in_channels should be satisfied
        s = torch.matmul(torch.transpose(hw_flatten(self.g(x)), 1, 2), hw_flatten(self.f(x)))

        beta = self.softmax(s)

        o = torch.matmul(beta, torch.transpose(hw_flatten(self.h(x)), 1, 2))
        o = torch.transpose(o, 1, 2)
        o = torch.reshape(o, x.shape)
        x = self.attention_multiplier * o + x
        return x


def down_sampling(x, covn_fn, pooling, out_channels, kernel, strides, update_collection, name, padding='same'):
    """
        Performs convolution plus downsamping if required
    Args:
      x: tensor input
      covn_fn: function that performs convulutions If False, the spatial size of the input tensor is unchanged.
      pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
      out_channels: Number of features in the output layer.
      kernel: The height and width of the convolution kernel filter
      strides: Rate of convolution strides
      update_collection: The update collections used in the in the spectral_normed_weight. (Default value = None)
      name:  Variable scope name
      padding:  Padding type (Default value = 'same')

    Returns:
        A tensor representing the output of the operation.
    """
    if pooling == 'avg':
        x = covn_fn(x, out_channels, kernel, strides, update_collection=update_collection, name=name, padding=padding)
        x = torch.nn.AvgPool2d(x, kernel, strides)
    else:
        x = covn_fn(x, out_channels, kernel, strides, update_collection=update_collection, name=name, padding=padding)
    return x

# Just use torch.nn.Upsample
#def up_sample(x, multipliers):
#    """
#        Upsamples given input
#    Args:
#      x: Input tensor
#      multipliers: A tuple of two ints. First one - factor by which to increase height.
#      Second one - factor by which to increase width.
#
#    Returns:
#        An upsampled tensor
#    """
#    x = torch.permute(x, (0, 3, 1, 2))
#    _, nx, nh, nw = x.shape
#    m = torch.nn.Upsample(x, scale_factor=multipliers, mode='nearest')
#    x = m(x)
#    return x

def _phase_shift(x, r, axis=1):
    """ Helper function with main phase shift operation

    Args:
      x: Tensor to upsample:
      r: Rate to upsample chosen axis
      axis: Axis to upsample (Default value = 1)

    Returns:
        Upsampled input
    """     

    bsize, c, a, b = x.shape
    assert c % r == 0
    remainder = int(c/r)
    X = torch.reshape(x, (-1, r, remainder, a, b))
    X = torch.split(X, r, dim=1) # a, [bsize, b, r, r]
    X = torch.concat([torch.squeeze(x) for x in X], dim=axis) # bsize, b, a * r, r

    if axis == 1:
        a = a * r
    elif axis == 2:
        b = b * r
    else:
        raise NotImplementedError
    return torch.reshape(X, (-1, a, b, remainder))


def upsample_ps(x, r=(1, 1)):
    """
    Performs phase shift upsampling for width and height
    Args:
      x: Tensor to upsample
      r: A tuple of upsample factors(height and width) (Default value = (1)
    Returns:
      Tensor with height of original_height*r[0] and width original_width*r[1]

    """
    x = _phase_shift(x, r[1], axis=2)
    x = _phase_shift(x, r[0], axis=1)
    return x

def cycle_padding(x, axis=1):
    """

    Args:
      x: Input tensor
      axis: The axis along which cycle padding needs to be applied (Default value = 1)
    Returns:
      A tensor that for chosen axis was padded with data from different end - creating a cycle

    """

    middle_point = x.shape[axis] // 2

    if axis == 1:
        prefix = x[:, middle_point + 1 :, :, :]
        suffix = x[:, :middle_point, :, :]
    elif axis == 2:
        prefix = x[:, :, :, middle_point + 1:]
        suffix = x[:, :, :, :middle_point]
    else:
        raise NotImplementedError

    return torch.concat([prefix, x, suffix], axis=axis)

def minibatch_stddev_layer(x, group_size=4):
    """
        Original version from ProGAN
    Args:
      x: Input tensor
      group_size:  The number of groups (Default value = 4)
    Returns:
        A standard deviation of chosen number groups. This result is repeated until the shape is matching input
        shape for concatication
    TODO (in original tf code):
        It contains bugs, however, it works better than fixed version. Needs some investigation
    """
    group_size = min(group_size, x.shape[0])
    s = x.shape #[NCHW] Input Shape.
    y = torch.reshape(x, [group_size, -1, s[1], s[2], s[3]]) #[GMHWC] Split minibatch into M groups of size G
    y = y.type(torch.float32) # [GMHWC] Cast to FP32.
    y -= torch.mean(y, dim=0, keepdim=True)  # [GMCHW] Subtract mean over group.
    y = torch.mean(torch.square(y), axis=0)  # [MCHW]  Calc variance over group.
    y = torch.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
    y = torch.mean(y, dim=(1, 2, 3), keepdim=True)  # [M111]  Take average over fmaps and pixels.
    y = y.type(x.dtype) # [M111]  Cast back to original data type.
    y = torch.tile(y, (group_size, 1, s[2], s[3]))  # [N1HW]  Replicate over group and pixels.
    return torch.concat([x, y], dim=1)  # [NHW]  Append as new fmap.

def minibatch_stddev_layer_v2(x, group_size=4):
    """
        Simplified version of ProGAN minibatch discriminator for 1D
    Args:
      x: Input tensor
      group_size:  The number of groups (Default value = 4)

    Returns:
        A standard deviation of chosen number groups. This result is repeated until the shape is matching input
        shape for concatication
    """
    group_size = min(group_size, x.shape[0])  # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape  # [NCHW]  Input shape.
    y = torch.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMHWC] Split minibatch into M groups of size G.
    y = y.type(torch.float32) # [GMHWC] Cast to FP32.
    y -= torch.mean(y, dim=0, keepdim=True)  # [GMCHW] Subtract mean over group.
    y = torch.mean(torch.square(y), axis=0)  # [MCHW]  Calc variance over group.
    y = torch.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
    y = torch.mean(y, dim=(1, 2, 3), keepdim=True)  # [M111]  Take average over fmaps and pixels.
    y = y.type(x.dtype) # [M111]  Cast back to original data type.
    y = torch.tile(y, (group_size, s[1], s[2], 1))  # [NHW1]  Replicate over group and pixels.
    return torch.concat([x, y], dim=3)  # [NHWC]  Append as new fmap.

def hw_flatten(x):
    """
        Flattens along the height and width of the tensor
    Args:
      x: Input tensor "NCHW"

    Returns:
        Flattened tensor
    """
    return torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))


def pad_up_to(x, output_shape, constant_values=0, dynamic_padding=False):
    """

    Args:
      x: Input tensor
      output_shape: Output shape
      constant_values:  Values used for padding (Default value = 0)

    Returns:
        Returns padded tensor that is the shape of output_shape.
    """
    s = list(x.size())

    # paddings_tf[i] -> (paddings[2 * i], paddings[2 * i + 1])
    paddings_tf = [calculate(s[i], m, dynamic_padding) for (i, m) in enumerate(output_shape)]
    paddings = []
    for pading in paddings_tf:
        paddings.append(pading[0])
        paddings.append(pading[1])

    return torch.nn.functional.pad(x, paddings, mode='constant', value=constant_values)

def calculate(current_shape, target_shape, dynamic_padding=False):
    if dynamic_padding:
        missing_padding = target_shape - current_shape

        def empty_padding(): return [missing_padding, missing_padding]

        def random_padding():
            front = torch.squeeze(torch.randint(size=1, high=missing_padding))
            end = missing_padding - front
            return [front, end]

        def middle_padding():
            front = torch.tensor(int(missing_padding/2), dtype=torch.int32)
            end = missing_padding - front
            return [front, end]

        # TODO: why is tf.cond or torch.eq is required
        if missing_padding ==  0:
            return empty_padding()
        else:
            return middle_padding()

    else:
        return [0, target_shape - current_shape]

def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + torch.erf(input_tensor / torch.sqrt(torch.tensor(2.0))))
    return input_tensor * cdf

def slerp(val, low, high):
    """
    Spherical linear interpolation
    Args:
        val: value
        low: lowest value
        high: highest value

    Returns:

    """
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def log(x, base):

    """
    Helper function to have a log function with chosen base
    Args:
        x: input on which to apply log with chosen base
        base: base for log

    Returns:
        input on which log with chosen base was applied
    """

    numerator = torch.log(x)
    denominator = torch.log(torch.tensor(base, dtype=numerator.dtype))
    return numerator/denominator

def get_padding(kernel, dilations, axis):
    """
    Calculates required padding for given axis
    Args:
        kernel: A tuple of kernel height and width
        dilations: A tuple of dilation height and width
        axis: 0 - height, 1 width
    Returns:
        An array that contains a length of padding at the begging and at the end
    """
    extra_padding = (kernel[axis] - 1) * (dilations[axis])
    return (extra_padding // 2, extra_padding - (extra_padding // 2))

def apply_padding(x, kernel, dilations, padding):
    """
    Adds padding to the edges of tensor such that after applying VALID conv the size of tensor remains the same.
    Acts similar to the same conv but instead of 0 padding it using REFLECT
    Args:
      x: Input tensor (NCHW)
      kernel: A tuple of kernel height and width
      dilations: A tuple of dilation height and width
      padding: Type of padding
    Returns:
        Padded tensor
    """
    height_padding = [0, 0]
    width_padding = [0, 0]
    apply = False
    if padding == 'VALID_H' or padding == 'VALID':
        height_padding = get_padding(kernel, dilations, 0)
        apply = True
    if padding == 'VALID_W' or padding == 'VALID':
        width_padding = get_padding(kernel, dilations, 1)
        apply = True
    if padding is None:
        padding = "VALID"
    if apply:
        pad = (*height_padding, *width_padding, 0, 0)
        x = F.pad(input = x, pad = pad, mode='reflect')
        padding = 'VALID'
    if padding == "VALID_ORIGINAL":
        padding = "VALID"
    return x, padding

class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 'same', stride=(1,1), dilation=(1,1)):
        """
            Creates convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple
        """
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.cn2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation = dilation, padding=padding)
        nn.init.xavier_uniform_(self.cn2d.weight)

    def forward(self, x):
        """ x -> NCHW """
        x, padding = apply_padding(x, self.kernel_size, self.dilation, self.padding)  
        return self.cn2d(x)

class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, stddev=0.02, init_bias=0):
        """
            Creates convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple or int
              stddev: The standard deviation for weights initializer. (Default value = 0.02)
              init_bias:   (Default value = 0.)
        """
        super().__init__()
        self.dcn2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation) 
        nn.init.normal_(self.dcn2d.weight, mean=init_bias, std=stddev)

    def forward(self, x):
        """ x -> NCHW """
        return self.dcn2d(x)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias_start=0.0):
        """
            Creates a linear layer.
            Args:
              in_features: 2D input tensor (batch size, features)
              out_features: Number of features in the output layer
              bias_start: The bias parameters are initialized to this value (Default value = 0.0)
            Returns:
            Linear transformation of input x
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias_start)

    def forward(self, x):
        """ x -> *, in_features """
        return self.linear(x)

def l2normalize(v, eps=1e-12):
    """l2 normalize the input vector.
    Args:
      v: tensor to be normalized
      eps:  epsilon (Default value = 1e-12)
    Returns:
      A normalized tensor
    """

    norm_v = F.normalize(v,dim=0,p=2)
    return norm_v

class snconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 'same', stride=(1,1), dilation=(1,1), sn_iters=1):
        """
            Creates a spectral normalized (SN) convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple
              sn_iters: number of power iterations ot calculate spectral norm
        """
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        non_sncn2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation = dilation)

        self.sncn2d = nn.utils.spectral_norm(non_sncn2d, n_power_iterations=sn_iters)
        nn.init.xavier_uniform_(self.sncn2d.weight)

    def forward(self, x):
        """ x -> NCHW """
        x, padding = apply_padding(x, self.kernel_size, self.dilation, self.padding)  
        return self.sncn2d(x)

class sndeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), dilation=(1,1), sn_iters=1):
        """
            Creates a spectral normalized (SN) convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple or int
              sn_iters: number of power iterations ot calculate spectral norm
        """
        super().__init__()
        non_sndcn2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation) 

        self.sndcn2d = nn.utils.spectral_norm(non_sndcn2d, n_power_iterations=sn_iters)
        nn.init.xavier_uniform_(self.sndcn2d.weight)

    def forward(self, x):
        """ x -> NCHW """
        return self.sndcn2d(x)

class snlinear(nn.Module):
    def __init__(self, in_features, out_features, bias_start=0.0, sn_iters=1):
        """
            Creates a spectral normalized linear layer.
            Args:
              in_features: 2D input tensor (batch size, features)
              out_features: Number of features in the output layer
              bias_start: The bias parameters are initialized to this value (Default value = 0.0)
              sn_iters: number of sn iterations 
            Returns:
            Linear transformation of input x
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.snlinear = nn.utils.spectral_norm(self.linear, n_power_iterations=sn_iters)

        nn.init.xavier_uniform_(self.snlinear.weight)
        nn.init.constant_(self.snlinear.bias, bias_start)

    def forward(self, x):
        """ x -> *, in_features """
        return self.snlinear(x)

class Sn_embedding(nn.Module):

    """ Creates a spectral normalized embedding lookup layer.
        Args:
            num_classes: The number of classes
            embedding_size: the lenght of embedding vector 
            sn_iters: number of SN iterations (default 1)
    """
    def __init__(self, number_classes, embedding_dim, sn_iters=1):
        super().__init__()
        embedding = nn.Embedding(number_classes, embedding_dim)
        self.sn_embedding = nn.utils.spectral_norm(embedding, n_power_iterations=sn_iters)

        nn.init.xavier_uniform_(self.sn_embedding.weight)

    def forward(self, x):
        """ x : * """
        return self.sn_embedding(x)

class scaled_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', stride=(1,1), dilation=(1,1), sn_iters=1):
        """
            Creates a scaled spectral normalized (SN) convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple or int
              sn_iters: number of power iterations ot calculate spectral norm
        """
        #BUG in org: spectral normalization is not performed in orignial code
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.cn = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation = dilation)

        wscale = get_scaled_weights(self.cn.weight.shape)
        nn.init.normal_(self.cn.weight, std=wscale)
    
    def forward(self, x):
        """ x -> NCHW """
        x, padding = apply_padding(x, self.kernel_size, self.dilation, self.padding)  
        return self.cn(x)

class scaled_deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), sn_iters=1):
        """
            Creates a scaled spectral normalized (SN) convolutional layers which use xavier initializer.
            Args:
              in_channels, out_channels: number of channels (int)
              kernel_size: size of kernel (tuple or int)
              stride: tuple or int
              dilation: tuple or int
              sn_iters: number of power iterations ot calculate spectral norm
        """
        super().__init__()
        self.cn = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation) 

        wscale = get_scaled_weights(self.cn.weight.shape)
        nn.init.normal_(self.cn.weight, std=wscale)

    def forward(self, x):
        """ x -> NCHW """
        return self.cn(x)

def get_scaled_weights(shape):
    """
    Get scaled weights
    Args:
        shape: the shape of required weight
    Returns:
        scaling factor
    """
    fan_in = np.prod(shape[:-1])
    std = np.sqrt(2) / np.sqrt(fan_in)  # He init
    wscale = torch.tensor(np.float32(std))

    return wscale

def pixel_norm(x, epsilon=1e-8):
    """
        Function that performs pixel norm
    Args:
        x: input
        epsilon:
    Returns:
        normalised input
    """
    sq = torch.square(x)
    return x * torch.rsqrt(sq.mean(axis=-1, keepdims=True) + epsilon)

class _block(nn.Module):
    def __init__(self, in_channels, out_channels, Conv=conv2d, kernel=(3, 3), stride = (2, 2), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same', batch_norm=False):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn0 = BatchNorm(out_channels)
            self.bn1 = BatchNorm(out_channels)

        self.conv = Conv(in_channels, out_channels, kernel_size=kernel, dilation = dilation, padding=padding)
        self.act = Act()
        self.pooling = pooling
        self.stride = stride

        if self.pooling=='avg':
            self.down_conv = Conv(out_channels, out_channels, kernel_size=kernel, padding=padding)
            self.avgPool2d = nn.AvgPool2d(stride, stride)
        else:
            self.down_conv = Conv(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)
        if stride[0] > 1 or stride[1] > 1 or self.in_channels != out_channels:
            if self.pooling=='avg':
                self.down_conv_0 = Conv(in_channels, out_channels, kernel_size=kernel, padding=padding)
                self.avgPool2d_0 = nn.AvgPool2d(stride, stride)
            else:
                self.down_conv_0 = Conv(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        x_0 = x
        x = self.conv(x)

        if self.batch_norm:
            x = self.bn0(x)

        x = self.act(x)

        if self.pooling=='avg':
            x = self.down_conv(x)
            x = self.avgPool2d(x)
        else:
            x = self.down_conv(x)

        if self.batch_norm:
            x = self.bn1(x)

        if self.stride[0] > 1 or self.stride[1] > 1 or self.in_channels != self.out_channels:
            if self.pooling=='avg':
                x_0 = self.down_conv_0(x_0)
                x_0 = self.avgPool2d_0(x_0)
            else:
                x_0 = self.down_conv_0(x_0)
        out = x_0 + x
        return out


class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride = (1, 1), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in GAN. It used standard 2D conv.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(in_channels, out_channels, conv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x)

class block_norm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same'):

        """Builds the residual blocks used in GAN. It used standard 2D conv and batch norm.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(in_channels, out_channels, conv2d, kernel, stride, dilation, Act, pooling, padding, True)

    def forward(self, x):
        return self.blk(x)

class sn_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in SNGAN. It used 2D conv with spectral normalization.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(in_channels, out_channels, snconv2d, kernel, stride, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x)

class sn_norm_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(1, 1), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in SNGAN. . It used 2D conv with spectral normalization and batchnorm
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections used in the spectral_normed_weight. (Default value = None)
          act: The activation function used in the block. (Default value = leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          downsample: If True, downsample the spatial size the input tensor.
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(in_channels, out_channels, snconv2d, kernel, stride, dilation, Act, pooling, padding, True)

    def forward(self, x):
        return self.blk(x)

class scaled_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), strides=(1, 1), dilation=(1, 1), Act=leaky_relu, pooling='avg', padding='same'):
        """Builds the residual blocks used in ProGAN. It used 2D conv with Equalized learning rate.
        Args:
          x: The 4D input vector.
          out_channels: Number of features in the output layer.
          name: The variable scope name for the block.
          kernel: The height and width of the convolution kernel filter (Default value = (3, 3))
          strides: The height and width of convolution strides (Default value = (1, 1))
          dilations: The height and width of convolution dilation (Default value = (1, 1))
          update_collection: The update collections. (Default value = None)
          act: The activation function used in the block. (Default value = leaky_relu)
          pooling: Strategy of pooling. Default: average pooling. Otherwise, no pooling, just using strides
          If False, the spatial size of the input tensor is unchanged. (Default value = True)
          padding:  Type of padding (Default value = 'same')
        Returns:
          A Tensor representing the output of the operation.
        """
        super().__init__()
        self.blk = _block(in_channels, out_channels, scaled_conv2d, kernel, strides, dilation, Act, pooling, padding, False)

    def forward(self, x):
        return self.blk(x)
