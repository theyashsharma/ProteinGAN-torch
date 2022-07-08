import torch
from torch import nn
from common.model import ops
from common.model.ops import sn_block, Self_Attn
from common.model.ops import leaky_relu, sn_block

class ResnetDiscriminator(nn.Module):
    def __init__(self, data, df_dim, shape, num_classes=None):
        super().__init__()
        self.act = leaky_relu()
        self.input_shape = shape
        self.kernel = (3, 3)
        self.pooling='avg'
        self.length = self.input_shape[1]
        self.height = self.input_shape[0]
        self.strides = self.get_strides()
        self.number_of_layers = len(self.strides)


        self.layers = []
        for layer_id in range(len(self.strides)):
            if layer_id == 1:
                self.atten = Self_Attn(data, df_dim, sn=True)
            dilation_rate = (1, 1)
            hidden_dim = df_dim * self.strides[layer_id][0]
            out_channels = hidden_dim // 2
            self.layers.append(sn_block(data, out_channels, self.kernel, self.strides[layer_id], padding='VALID'))        
        
        self.snlinear = ops.snlinear(self.length, 1)
        self.sigmoid = nn.Sigmoid()

    def get_strides(self):
        strides = [(1, 1), (1, 1), (1, 1), (1,1)]
        if self.length == 512:
            strides.extend([(1, 1), (1, 1)])
        return strides

    def forward(self, z):
        if len(z.shape) == 3:
            z = torch.unsqueeze(z, 1)
        for layer_id in range(self.number_of_layers):
            if layer_id == 1:
                pass
            out = self.layers[layer_id](z)
        
        out = self.act(out)  
        out = ops.minibatch_stddev_layer(out)  
        out = torch.sum(out, [1, 2])
        out = self.snlinear(out)
        out = self.sigmoid(out)
        
        return out

class GumbelDiscriminator(nn.Module):
    def __init__(self, df_dim, batch_size, shape, num_classes=None):
        super().__init__()

        self.dim = df_dim
        self.act = leaky_relu
        self.input_shape = shape
        self.length = self.input_shape[2]
        self.height = self.input_shape[1]
        self.output_shape = [batch_size, 1]
        self.strides = [(1, 2), (1, 2), (1, 2), (1, 2)]
        if self.length == 512:
            self.strides.extend([(1, 2), (1, 2)])

    def forward(self, data):
        if len(z.shape) == 3:
            z = torch.unsqueeze(z, 1)

        # Embedding
        embeddings = nn.Embedding(21, self.dim)
        h = self.embeddings(data)

        # Resnet
        hidden_dim = self.dim
        for layer in range(len(self.strides)):
            self.log(h.shape)
            block_name, dilation_rate, hidden_dim, strides = self.get_block_params(hidden_dim, layer)
            h = self.add_sn_block(h, hidden_dim, block_name, dilation_rate, strides)
            if layer == 0:
                self.add_attention(h, hidden_dim)

        end_block = self.act(h)
        h_std = ops.minibatch_stddev_layer_v2(end_block)

        final_conv = ops.snconv2d(h_std, int(hidden_dim / 16), (1, 1), name='final_conv', padding=None)
        self.log(final_conv.shape)
        output = ops.snlinear(torch.squeeze(nn.Flatten(final_conv)), 1)
        nn.Flatten
        return output, end_block

