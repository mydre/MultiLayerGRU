"""main.py"""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch

from solver import Solver
from utils.utils import str2bool
from bottleneck_transformer_pytorch import BottleStack

def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = Solver(args)

    if args.mode == 'train':
        net.train(args)
    elif args.mode == 'test':
        net.test()
    elif args.mode == 'gene_label': # 产生预测的标签值和实际的标签值
        net.gene_label()
    elif args.mode == 'gene_probability': # 生成画ROC曲线时会用到了原材料,即实际的标签值，即各个类别对应的softmax列向量
        net.gene_probability()
    elif args.mode == 'wgan': # 训练wgan模型
        net.gan(args.filename)
    elif args.mode == 'enhancement': # mnist向量输入到wgan模型的生成器中进行训练接样本增强
        net.ganSamplerEnhancement(args.filename)
    elif args.mode == 'merge_data':
        net.merge_data()
    else: return

    print('[*] Finished')

'''
layer = BottleStack(
    dim = 1,
    fmap_size = 43,
    dim_out = 8,
    proj_factor = 4,
    downsample = True,
    heads = 4,
    dim_head = 8,
    rel_pos_emb = False,
    activation = nn.ReLU()
)

fmap = torch.randn(2,1,43,43)
print(fmap.shape)
layer(fmap)
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='hahahahhahahhh')
    parser.add_argument('--epoch', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    # gan模型训练的时候的learning rate是2e-6
    # parser.add_argument('--lr', type=float, default=2e-6, help='learning rate')
    parser.add_argument('--y_dim', type=int, default=2, help='the number of classes')
    parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    parser.add_argument('--env_name', type=str, default='main', help='experiment name')
    parser.add_argument('--eps', type=float, default=1e-9, help='epsilon')
    parser.add_argument('--dataset', type=str, default='FMNIST', help='dataset type')
    parser.add_argument('--pixel_width', type=int, default=19, help='the width of minist data picture')
    parser.add_argument('--dset_dir', type=str, default='datasets', help='dataset directory path')
    parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    parser.add_argument('--load_ckpt', type=str, default='', help='')
    parser.add_argument('--cuda', type=str2bool, default=True, help='enable cuda')
    parser.add_argument('--silent', type=str2bool, default=False, help='')
    parser.add_argument('--mode', type=str, default='train', help='train / test / generate / universal / gene_label / gene_probability')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    parser.add_argument('--tensorboard', type=str2bool, default=False, help='enable tensorboard')
    parser.add_argument('--visdom', type=str2bool, default=False, help='enable visdom')
    parser.add_argument('--visdom_port', type=str, default=55558, help='visdom port')
    parser.add_argument('--desc', type=str, default='', help='the describe of generated Label')
    parser.add_argument('--filename', type=str, default='', help='the filename of wgan model')
    args = parser.parse_args()

    main(args)
