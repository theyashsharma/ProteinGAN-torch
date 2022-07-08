import torch
from gan.sngan_.discriminator import ResnetDiscriminator
from gan.sngan_.generator import ResnetGenerator

model = ResnetGenerator(100, 10, (512, 21))
z = torch.zeros((64, 100))
p = model.forward(z)
print('here')
print(z.shape)
print(p.shape)
#for param in model.parameters():
#    print(param)

#model = ResnetDiscriminator(100, 10, 16, (512, 21))
#data = torch.zeros((64, 100))
#p = model.forward(data)
#print('here')
#print(data.shape)
#print(p.shape)
##for param in model.parameters():
##    print(param)



"""
class ResnetGenerator(nn.Module):
    def __init__(self, gf_dim, shape, num_features, num_classes=None):
        super().__init__()

        self.dim = gf_dim
        self.act = leaky_relu
        self.output_shape = shape
        self.kernel = (3, 3)
        self.pooling='avg'
        self.length = self.output_shape[2]
        self.height = self.output_shape[1]
        self.strides = self.get_strides()

        for s in self.strides:
            self.width_d = 1 * s[1]
            self.height_d = 1 * s[0]

        self.number_of_layers = len(self.strides)
        self.hidden_dim = self.dim * (2 ** (self.number_of_layers-1))
        self.c_h = int(self.height / self.height_d)
        self.c_w = int((self.length / self.width_d))

        for layer_id in range(self.number_of_layers):
            dilation_rate = (1,1)
            self.h = sn_block(10, self.hidden_dim, self.kernel, self.strides[layer_id], dilation_rate,
                            self.act, self.pooling, 'VALID')
            if layer_id == self.number_of_layers - 2:
                self.h = Self_Attn(10, self.hidden_dim, sn=True)
            self.hidden_dim = self.hidden_dim / self.strides[layer_id][1]
            
        self.num_features = num_features
        self.number_of_layers = len(self.strides)
        self.starting_dim = self.dim * (2 ** self.number_of_layers)

        self.bn = ops.BatchNorm()
        self.h_act = leaky_relu(self.bn(self.h))

        if self.output_shape[2] == 1:
            self.out = nn.Tanh(ops.snconv2d(self.h_act, 1, (self.output_shape[0], 1)))
        else:
            self.out = nn.Tanh(ops.snconv2d(self.h_act, 21, (1, 1)))

    def get_strides(self):
        strides = [(1, 2), (1, 2), (1, 2), (1,2)]
        if self.length == 512:
            strides.extend([(1, 2), (1, 2)])
        return strides

    def forward(self, z):
        final_bn = ops.BatchNorm(self.num_features)
        h = ops.snlinear(z, self.c_h * self.c_w * self.hidden_dim)
        h = torch.reshape(h, [-1, self.hidden_dim, self.c_h, self.c_w ])
        h = self.h(h)
        bn = ops.BatchNorm()
        h_act = leaky_relu(bn(h))
        out = self.out(h_act)
        return out
"""
