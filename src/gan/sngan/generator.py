import torch
from torch import nn
from common.model.generator_ops import sn_block
from common.model.ops import leaky_relu, log
from common.model import ops
from common.model.ops import Self_Attn
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

class ResnetGenerator(nn.Module):
    def __init__(self, zs_dim, gf_dim, shape, num_classes=None):
        super().__init__()

        self.dim = gf_dim
        self.act = leaky_relu
        self.output_shape = shape
        self.length = self.output_shape[1]
        self.height = self.output_shape[0]
        self.strides = self.get_strides()
        self.kernel = (3, 3)
        self.pooling='avg'

        self.width_d = 1
        self.height_d = 1
        for s in self.strides:
            self.width_d = 1 * s[1]
            self.height_d = 1 * s[0]

        self.number_of_layers = len(self.strides)
        self.starting_dim = self.dim * (2 ** self.number_of_layers)

        self.c_h = int(self.height / self.height_d)
        self.c_w = int((self.length / self.width_d))
        self.hidden_dim = self.dim * (2 ** (self.number_of_layers - 1))

        self.org_hidden = self.hidden_dim
        self.sn_linear = ops.snlinear(zs_dim, self.c_h * self.c_w * self.hidden_dim)

        self.layers = []
        for layer_id in range(self.number_of_layers):
            dilation_rate = (1, 1)
            out_channels = self.hidden_dim // 2
            self.layers.append(sn_block(self.hidden_dim, out_channels, self.kernel, self.strides[layer_id], padding='VALID'))
            self.hidden_dim = out_channels

            if layer_id == self.number_of_layers - 2:
                self.atten = Self_Attn(self.hidden_dim, self.hidden_dim, sn=True)

        
        self.act_stk = nn.Sequential(
            ops.BatchNorm(self.hidden_dim),
            leaky_relu(),
            ops.snconv2d(self.hidden_dim, 1, (1, 1))
        )

    def get_strides(self):
        strides = [(1, 1), (1, 1), (1, 1), (1,1)]
        if self.length == 512:
            strides.extend([(1, 1), (1, 1)])
        return strides

    def forward(self, z):

        out = self.sn_linear(z)
        out = torch.reshape(out, (-1, self.org_hidden, self.c_h, self.c_w))
        print(out.shape)

        for layer_id in range(self.number_of_layers):
            out = self.layers[layer_id](out)

            if layer_id == self.number_of_layers - 2:
                # TODO: 512 * 21 is too large for matrix mul
                pass

        out = self.act_stk(out)
        print(out.shape)

        return out

class GumbelGenerator(nn.Module):
    def __init__(self, gf_dim, shape, num_features, num_classes=None):
        super().__init__()

        self.dim = gf_dim
        self.act = leaky_relu
        self.output_shape = shape
        self.length = self.output_shape[2]
        self.height = self.output_shape[1]
        self.strides = self.get_strides()
        self.num_features = num_features
        self.number_of_layers = len(self.strides)
        self.starting_dim = self.dim * (2 ** self.number_of_layers)

    def get_strides(self):
        strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        if self.length == 512:
            strides.extend([(1, 2), (1, 2)])
        return strides

    def forward(self, z):

        # Fully connected
        i_shape = self.initial_shape
        h = ops.snlinear(z, i_shape[1] * i_shape[2] * i_shape[3], name='noise_linear')
        h = torch.reshape(h, i_shape)

        # Resnet architecture
        hidden_dim = self.starting_dim
        for layer_id in range(self.number_of_layers):
            self.log(h.shape)
            block_name, dilation_rate, hidden_dim, stride = self.get_block_params(hidden_dim, layer_id)
            h = self.add_sn_block(h, hidden_dim, block_name, dilation_rate, stride)
            if layer_id == self.number_of_layers - 2:
                h = self.add_attention(h, hidden_dim)
                hidden_dim = hidden_dim*2

        # Final conv
        h_act = self.act(self.final_bn(h), name="h_act")
        last = ops.snconv2d(h_act, 21, (1, 1), name='last_conv')

        # Gumbel max trick
        out = RelaxedOneHotCategorical(temperature=self.get_temperature(True), logits=last).sample()
        return out

