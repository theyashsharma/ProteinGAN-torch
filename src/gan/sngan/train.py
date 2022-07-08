import torch 
import torch.nn as nn
import numpy as np
from gan.sngan.model import SNGAN
from gan.data.dataLoader import get_dataLoader
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet', help="type of model used")
    parser.add_argument("--shape", type=list, default=[5, 5, 3], help="shape")
    parser.add_argument("--shaped", type=list, default=(512, 21), help="shape")
    parser.add_argument("--num_features", type=int, default=10, help="number of features")
    parser.add_argument("--noise_dim", type=int, default=100, help="latent embeddings size")
    parser.add_argument("--epochs", type=int, default=50, help="rounds of training")
    parser.add_argument("--df_dim", type=int, default=10, help="df dimentions")
    parser.add_argument("--gf_dim", type=int, default=10, help="gf_dimentions")
    parser.add_argument("--zs_dim", type=int, default=100, help="zs_dimentions")
    parser.add_argument("--data", type=int, default=1, help="zs_dimentions")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    if args.model == 'resnet':
        gen_arguments = {'zs_dim': args.zs_dim,'gf_dim': args.gf_dim, 'shape': args.shaped, 'num_classes': None}
        disc_arguments = {'data': args.data, 'df_dim': args.df_dim, 'shape': args.shaped, 'num_classes': None}
    elif args.model == 'gumbel':
        disc_arguments = {'df_dim': args.df_dim, 'batch_size' : args.batch_size, 'shape': args.shape, 'num_classes': None}
        gen_arguments = {'gf_dim': args.gf_dim, 'shape': args.shape, 'num_features': args.num_features, 'num_classes': None}

    sngan = SNGAN(args.model,  generator_args = gen_arguments, discriminator_args = disc_arguments)

    dataLoader = get_dataLoader('../data/', 64)
    criterion = torch.nn.BCELoss()

    sngan.train(dataLoader, args.lr, args.epochs, criterion)
