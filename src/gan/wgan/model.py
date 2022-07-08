from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from gan.gan_base_model import GAN
from gan.wgan.discriminator import discriminator_fully_connected, original_discriminator, discriminator_resnet
from gan.wgan.generator import generator_fully_connected, original_generator, generator_resnet

class WGAN:
    """WGAN model."""
    def __init__(self, arch, generator_args, discriminator_args):
        if arch == 'fully_connected':
            self.generator = generator_fully_connected(**generator_args)
            self.discriminator = discriminator_fully_connected(**discriminator_args)
        elif arch == 'original':
            self.generator = original_generator(**generator_args)
            self.discriminator = original_discriminator(**discriminator_args)
        elif arch == 'resnet':
            self.generator = generator_resnet(**generator_args)
            self.discriminator = discriminator_resnet(**discriminator_args)

    def train(self, dataLoader, lr, num_epochs, criterion):

        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr = lr)
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr = lr)

        losses_G = []
        losses_D = []

        for epoch in range(num_epochs):

            running_D_loss = 0
            running_G_loss = 0

            for index, real_X in enumerate(dataLoader):

                # TODO device
                labels = torch.ones(real_X.shape[0], 1)
                self.discriminator.zero_grad()
                real_X = real_X.type(torch.float32)

                # real forward
                output = self.discriminator(real_X)
                error_D_real = criterion(output, labels)
                error_D_real.backward()
                D_x = output.mean().item()

                # fake forward
                labels_fake = torch.zeros(real_X.shape[0], 1)
                noise = torch.randn(real_X.shape[0], 100)
                fake_data = self.generator(noise)
                output = self.discriminator(fake_data.detach().squeeze())

                error_D_fake = criterion(output, labels_fake)
                error_D_fake.backward()
                D_G_z1 = output.mean().item()

                optimizer_D.step()

                # generator
                labels.fill_(1)
                self.generator.zero_grad()

                output = self.discriminator(fake_data)
                error_G = criterion(output, labels)
                D_G_z2 = output.mean().item()
                error_G.backward()

                optimizer_G.step()

                error_D = error_D_real + error_D_fake
                running_D_loss += error_D.item()
                running_G_loss += error_G.item()


                if index % 1 == 0:
                    print(f'epoch {epoch + 1}, index {index + 1}: g_loss={running_G_loss:.4f}, d_loss={running_D_loss:.4f}')        
                    print(f'D(x): {D_x:.3f}, D(G(z)): {D_G_z1:.3f} {D_G_z2:.3f}')
                    print()

        losses_G.append(running_G_loss)
        losses_D.append(running_D_loss)

        return losses_G, losses_D
