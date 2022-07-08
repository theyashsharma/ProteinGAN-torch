import torch 
import torch.nn as nn
import numpy as np
from gan.wgan.discriminator import discriminator_fully_connected, original_discriminator, discriminator_resnet
from gan.wgan.generator import generator_fully_connected, original_generator, generator_resnet
from gan.wgan.model import WGAN
from gan.data.dataLoader import get_dataLoader
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='fully_connected', help="type of model used")
    parser.add_argument("--latent_size", type=int, default=250, help="latent embeddings size")
    parser.add_argument("--channels", type=int, default=10, help="latent embeddings size")
    parser.add_argument("--noise_dim", type=int, default=100, help="latent embeddings size")
    parser.add_argument("--epochs", type=int, default=50, help="rounds of training")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    
    if args.model == 'fully_connected':
        gen_arguments = {'input_dim': args.noise_dim, 'labels': None, 'gf_dim': args.latent_size, 'num_classes': None}
        disc_arguments = {'in_shape': 512 * 21, 'labels': None, 'df_dim': args.channels, 'number_classes': None}
    elif args.model == 'original':
        disc_arguments = {'in_shape': 1, 'labels': None, 'df_dim': args.channels, 'number_classes': None, 'width' : 512, 'height' : 21}
        gen_arguments = {'zs_dim': args.noise_dim, 'labels': None, 'gf_dim': 10, 'num_classes': None}
    else:
        disc_arguments = {'in_shape': 1, 'labels': None, 'df_dim': args.channels, 'number_classes': None, 'width' : 512, 'height' : 21}
        gen_arguments = {'zs_dim': args.noise_dim, 'labels': None, 'gf_dim': 10, 'num_classes': None}

    wgan = WGAN(args.model, gen_arguments, disc_arguments)

    dataLoader = get_dataLoader('../data/', 64)
    criterion = torch.nn.BCELoss()

    wgan.train(dataLoader, args.lr, args.epochs, criterion)
    

    
    

