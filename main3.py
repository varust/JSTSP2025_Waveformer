# coding:utf-8

import torch
from torch.backends import cudnn
import numpy as np
import os
import time
import argparse
from train import _train
from utils import check_dir, print2txt, data_parallel, get_data_dir, gen_log
from architecture import *
import datetime
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename
def args2str(args, indent_l=1):
    """args to string for logger"""
    msg = ''
    for k, v in vars(args).items():
        msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    msg += '\n'
    return msg

def main(args):
    cudnn.benchmark = True
    model = model_generator(args.method,args)

    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    args.result_path = args.outf + date_time + f'S{args.method}/result/'
    args.model_path = args.outf + date_time + f'S{args.method}/model/'
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    #model = build_net(args)
    # print(model)
    logger = gen_log(args.model_path)
    logger.info("Loggers: {}".format(args2str(args)))
    logger.info("Learning rate:{}, batch_size:{}.\n".format(args.lr, args.batch_size))
    logger.info("===> Printing model\n{%s}" % (model))
    params = sum(param.numel() for param in model.parameters())
    logger.info('params: %s' % params)
    if args.device == 'cuda':
        #gpu_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        #model = data_parallel(model, gpu_num)

    # ToDO: model-ema
    _train(model, args, logger)



if __name__ == '__main__':
    base_dir = os.path.split(os.path.realpath(__file__))[0]
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--gpu', type=str, default='1,2')
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument("--num_trainset", type=int, default=5000)
    parser.add_argument("--num_channel", type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--patch_size", type=tuple, default=(256, 256))
    parser.add_argument("--test_size", type=tuple, default=(480, 512))
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--point_epoch', '-p', type=int, default=1)
    parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    parser.add_argument('--outf', type=str, default='./exp/', help='saving_path')
    parser.add_argument('--resuming_model_path', type=str, default='/home/lkshpc/liumengzu/Code/MSFA/memory/exp/2025_05_14_00_57_12Sspa_spe_var2/model/model_341.pth', help='saving_path')
    parser.add_argument("--resume_epoch", type=int, default=1)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--noise_level", type=int, default=0)
    parser.add_argument("--eta_min",type=float,default=1e-7)
    # Model parameters
    parser.add_argument('--model_name', type=str, default='mog-dun')  # , choices=['mog-dun', 'MIMO-UNetPlus'])
    parser.add_argument('--method', type=str, default='spa_spe_var4')
    parser.add_argument('--dataset', type=str, default='NTIRE')
    parser.add_argument("--T", type=int, default=4)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')  # fusedlamb
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--step', type=float, default=300, metavar='LR', help='learning rate (default: 5e-4)')

    # Path parameters
    parser.add_argument('--train_image_dir', type=str, default='/data/liumengzu/data/Dataset/ARAD_training/')
    parser.add_argument('--valid_image_dir', type=str, default='/data/liumengzu/data/Dataset/ARAD_testing/')

    parser.add_argument('--msfa_path', type=str, default='/data/liumengzu/data/Dataset/MSFA_16.mat')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--result_path', type=str, default='')

    args = parser.parse_args()

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'cuda':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # torch.cuda.manual_seed(args.seed)

    main(args)
