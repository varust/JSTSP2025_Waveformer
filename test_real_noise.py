# coding:utf-8


import torch
import torch.utils.data as tud
import os
import time
import datetime
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
# from dataset import Dataset
from utils import gen_mask,prepare_NTIRE_data, find_last_checkpoint, print2txt, rearrange_channel, save_image, compare_psnr,torch_psnr,torch_ssim,compute_psnr,load_img
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as sio
import argparse
from architecture import *
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import hdf5storage
def mask_input(GT_image, msfa_size):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], msfa_size ** 2), dtype=np.float32)
    for i in range(0,msfa_size):
        for j in range(0,msfa_size):
            mask[i::msfa_size, j::msfa_size, i*msfa_size+j] = 1

    input_image = mask * GT_image
    return input_image
def reorder_2filter(old):
    ###从波段中心从小到大的排列，reorder为滤波器从左往右依次的顺序（也就是GT图中的顺序）
    ### reorder the multiband cube as the real pattern in MSFA
    _,_,C = old.shape
    new = np.zeros_like(old)
    if C == 16:
        order = [2, 0, 9, 1, 15, 14, 12, 13, 7, 6, 4, 5, 11, 10, 8, 3]
        for i in range(0, 16):
            new[ :, :, order[i]] = old[ :, :, i]
        return new
def reorder_imecNtire(old):
    ### reorder the multiband cube, making the center wavelength from small to large
    # NTIRE small to large
    _,_,C = old.shape
    new = np.zeros_like(old)
    if C == 16:#
        new[:, :, 0] = old[:, :, 12]
        new[:, :, 1] = old[:, :, 8]
        new[:, :, 2] = old[:, :, 13]
        new[:, :, 3] = old[:, :, 9]
        new[:, :, 4] = old[:, :, 14]
        new[:, :, 5] = old[:, :, 10]
        new[:, :, 6] = old[:, :, 15]
        new[:, :, 7] = old[:, :, 11]
        new[:, :, 8] = old[:, :, 4]
        new[:, :, 9] = old[:, :, 0]
        new[:, :, 10] = old[:, :, 5]
        new[:, :, 11] = old[:, :, 1]
        new[:, :, 12] = old[:, :, 6]
        new[:, :, 13] = old[:, :, 2]
        new[:, :, 14] = old[:, :, 7]
        new[:, :, 15] = old[:, :, 3]
    return new
def test(img_lq, model, args, window_size=8,mask=None):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, 16, h*sf, w*sf).type_as(img_lq)
        Em = torch.zeros(b, 16, h * sf, w * sf).type_as(mask)
        W = torch.zeros_like(E)
        #Wm = torch.zeros_like(Em)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                Em_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch,Em_patch)
                Xtx = in_patch.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./input.png")
                Xtx = out_patch.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./out.png")
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def _test(args):
    model = model_generator(args.method, args)
    model = torch.nn.DataParallel(model, device_ids=[0])
    if args.resume:
        models1_pretrain = torch.load(os.path.join(args.resuming_model_path), map_location='cpu')  # .cuda()
        model.load_state_dict(models1_pretrain['model'], strict=True)

    args.mode = 'test'
    model.eval()
    if args.test_set == 'MCAN_REAL':
        size = (1024, 2048)
        mask = gen_mask(torch.from_numpy(sio.loadmat(args.msfa_path)['msfa']).permute(2, 0, 1),
                        (args.num_channel, size[0], size[1]), args.device)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(args.device).unsqueeze(0)
        # self.mask = self.mask.transpose(1, 2, 0)
        image_filenames = [join(args.test_image_dir, x) for x in sorted(listdir(args.test_image_dir))]
        real_result = []
        real_input = []
        for index in range(4):
            input = load_img(image_filenames[index])
            input = np.fliplr(input)
            input = input.astype(np.float32) / 255.  # 16 512 512
            input = torch.from_numpy(input)
            input = input.to(args.device).unsqueeze(0)
            print(input.dtype, mask.dtype)
            print(input.shape, mask.shape)
            with torch.no_grad():
                pred = test(input,model,args,mask=mask)
                #pred = model(input, mask)
                real_input.append(input.detach().cpu().numpy())
                real_result.append(pred.detach().cpu().numpy())
        name_realimg = args.result_path + '/realbest.mat'
        sio.savemat(name_realimg, {'input': real_input, 'pred': real_result})
        print("OK!!!")
    if args.test_set == 'KAIST':
        size = (256, 256)
        mask = gen_mask(torch.from_numpy(sio.loadmat(args.msfa_path)['msfa']).permute(2, 0, 1),
                        (args.num_channel, size[0], size[1]), args.device)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(args.device).unsqueeze(0)
        # self.mask = self.mask.transpose(1, 2, 0)
        image_filenames = [join(args.test_image_dir, x) for x in sorted(listdir(args.test_image_dir))]
        image_filenames.sort()
        real_result = []
        real_input = []
        label_result = []
        psnr_list = []
        ssim_list = []
        for index in range(10):
            label = sio.loadmat(image_filenames[index])['data']
            label = label.astype(np.float32)
            label = reorder_2filter(label)
            input = mask_input(label, 4)  # genereate real mosaic
            #input_image = reorder_imecMCAN(input_image)
            raw = input.sum(axis=2)
            raw = np.expand_dims(raw, axis=0)

            input = torch.from_numpy(raw)
            input = input.to(args.device).unsqueeze(0)
            label = torch.from_numpy(label).permute(2,0,1).to(args.device).unsqueeze(0)
            print(input.dtype, mask.dtype)
            print(input.shape, mask.shape)
            with torch.no_grad():
                #pred = test(input,model,args,mask=mask)
                pred = model(input, mask)
                Xtx = input.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./input.png")
                Xtx = pred.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./out.png")
                real_input.append(input.detach().cpu().numpy())
                real_result.append(pred.detach().cpu().numpy())
                label_result.append(label.cpu().numpy())
                for k in range(label.shape[0]):
                    # psnr_val_v = compute_psnr(result[k, :, :, :].detach().cpu().numpy(), label[k, :, :, :].detach().cpu().numpy(), 1)
                    psnr_val = torch_psnr(pred[k, :, :, :], label[k, :, :, :])
                    ssim_val = torch_ssim(pred[k, :, :, :], label[k, :, :, :])

                    # psnr_list_v.append(psnr_val_v)
                    psnr_list.append(psnr_val.detach().cpu().numpy())
                    ssim_list.append(ssim_val.detach().cpu().numpy())

        psnr_mean = np.mean(np.asarray(psnr_list))
        ssim_mean = np.mean(np.asarray(ssim_list))
        name_realimg = args.result_path + f'/{args.test_set}realbest.mat'
        sio.savemat(name_realimg, {'input': real_input, 'pred': real_result, 'label': label_result, 'psnr_list': psnr_list, 'ssim_list': ssim_list})
        print(f"PSNR: {psnr_mean}, SSIM: {ssim_mean}")
        print("OK!!!")
    if args.test_set == 'ICVL':
        size = (1300, 1392)
        mask = gen_mask(torch.from_numpy(sio.loadmat(args.msfa_path)['msfa']).permute(2, 0, 1),
                        (args.num_channel, size[0], size[1]), args.device)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(args.device).unsqueeze(0)
        # self.mask = self.mask.transpose(1, 2, 0)
        image_filenames = [join(args.test_image_dir, x) for x in sorted(listdir(args.test_image_dir))]
        image_filenames.sort()
        real_result = []
        real_input = []
        label_result = []
        psnr_list = []
        ssim_list = []
        for index in range(16):
            print(len(image_filenames))
            label = sio.loadmat(image_filenames[index])['data']
            label = label.astype(np.float32)
            label = reorder_2filter(label)
            input = mask_input(label, 4)  # genereate real mosaic
            # input_image = reorder_imecMCAN(input_image)
            raw = input.sum(axis=2)
            raw = np.expand_dims(raw, axis=0)

            input = torch.from_numpy(raw)
            input = input.to(args.device).unsqueeze(0)
            label = torch.from_numpy(label).permute(2,0,1).to(args.device).unsqueeze(0)
            print(input.dtype, mask.dtype)
            print(input.shape, mask.shape)
            with torch.no_grad():
                pred = test(input,model,args,mask=mask)
                #pred = model(input, mask)
                Xtx = input.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./input.png")
                Xtx = pred.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./out.png")
                real_input.append(input.detach().cpu().numpy())
                real_result.append(pred.detach().cpu().numpy())
                label_result.append(label.cpu().numpy())
                for k in range(label.shape[0]):
                    # psnr_val_v = compute_psnr(result[k, :, :, :].detach().cpu().numpy(), label[k, :, :, :].detach().cpu().numpy(), 1)
                    psnr_val = torch_psnr(pred[k, :, :, :], label[k, :, :, :])
                    ssim_val = torch_ssim(pred[k, :, :, :], label[k, :, :, :])

                    # psnr_list_v.append(psnr_val_v)
                    psnr_list.append(psnr_val.detach().cpu().numpy())
                    ssim_list.append(ssim_val.detach().cpu().numpy())

        psnr_mean = np.mean(np.asarray(psnr_list))
        ssim_mean = np.mean(np.asarray(ssim_list))
        name_realimg = args.result_path + f'/{args.test_set}realbest.mat'
        sio.savemat(name_realimg, {'input': real_input, 'pred': real_result, 'label': label_result, 'psnr_list': psnr_list, 'ssim_list': ssim_list})
        print(f"PSNR: {psnr_mean}, SSIM: {ssim_mean}")
        print("OK!!!")
    if args.test_set == 'NTIRE':
        size = (480, 512)
        mask = gen_mask(torch.from_numpy(sio.loadmat(args.msfa_path)['msfa']).permute(2, 0, 1),
                        (args.num_channel, size[0], size[1]), args.device)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(args.device).unsqueeze(0)
        # self.mask = self.mask.transpose(1, 2, 0)
        image_filenames = [join(args.test_image_dir, x) for x in sorted(listdir(args.test_image_dir))]
        image_filenames.sort()
        real_result = []
        real_input = []
        label_result = []
        psnr_list = []
        ssim_list = []
        for index in range(50):
            label = hdf5storage.loadmat(image_filenames[index])['cube']
            label = label.astype(np.float32)
            label = reorder_imecNtire(label)
            label = reorder_2filter(label)
            input = mask_input(label, 4)  # genereate real mosaic
            # input_image = reorder_imecMCAN(input_image)
            raw = input.sum(axis=2)
            raw = np.expand_dims(raw, axis=0)

            input = torch.from_numpy(raw)
            input = input.to(args.device).unsqueeze(0) + torch.randn_like(input).to(args.device)*args.noise_level/255
            label = torch.from_numpy(label).permute(2,0,1).to(args.device).unsqueeze(0)
            print(input.dtype, mask.dtype)
            print(input.shape, mask.shape)
            with torch.no_grad():
                #pred = test(input,model,args,mask=mask)
                pred = model(input, mask)
                Xtx = input.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./input.png")
                Xtx = pred.permute(2, 3, 0, 1).cpu().numpy()
                plt.imshow(Xtx[:, :, 0, 0])#, cmap='hot')
                plt.savefig(f"./out.png")
                real_input.append(input.detach().cpu().numpy())
                real_result.append(pred.detach().cpu().numpy())
                label_result.append(label.cpu().numpy())
                for k in range(label.shape[0]):
                    # psnr_val_v = compute_psnr(result[k, :, :, :].detach().cpu().numpy(), label[k, :, :, :].detach().cpu().numpy(), 1)
                    psnr_val = torch_psnr(pred[k, :, :, :], label[k, :, :, :])
                    ssim_val = torch_ssim(pred[k, :, :, :], label[k, :, :, :])

                    # psnr_list_v.append(psnr_val_v)
                    psnr_list.append(psnr_val.detach().cpu().numpy())
                    ssim_list.append(ssim_val.detach().cpu().numpy())

        psnr_mean = np.mean(np.asarray(psnr_list))
        ssim_mean = np.mean(np.asarray(ssim_list))
        name_realimg = args.result_path + f'/{args.test_set}realbest.mat'
        sio.savemat(name_realimg, {'input': real_input, 'pred': real_result, 'label': label_result, 'psnr_list': psnr_list, 'ssim_list': ssim_list})
        print(f"PSNR: {psnr_mean}, SSIM: {ssim_mean}")
        print("OK!!!")
base_dir = os.path.split(os.path.realpath(__file__))[0]
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--gpu', type=str, default='7')
parser.add_argument('--num_worker', type=int, default=4)
parser.add_argument("--num_trainset", type=int, default=5000)
parser.add_argument("--num_channel", type=int, default=16)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument("--patch_size", type=tuple, default=(256, 256))
parser.add_argument("--test_size", type=tuple, default=(480, 512))
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--point_epoch', '-p', type=int, default=1)
parser.add_argument('--mode', '-m', type=str, default='train', choices=['train', 'valid', 'test'])
parser.add_argument("--seed", type=int, default=1)
parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
parser.add_argument('--outf', type=str, default='./exp/', help='saving_path')

# last_method
# parser.add_argument('--method', type=str, default='transRWKVb1dct')
# parser.add_argument('--resuming_model_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_10_13_10_43StransRWKVb1dct/model/model_341.pth', help='saving_path')
# parser.add_argument('--result_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_10_13_10_43StransRWKVb1dct/model/', help='saving_path')
# final_method
# parser.add_argument('--method', type=str, default='transRWKVb1dctv1')
# parser.add_argument('--resuming_model_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_16_16_49_46StransRWKVb1dctv1_noise0/model/model_365.pth', help='saving_path')
# parser.add_argument('--result_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_16_16_49_46StransRWKVb1dctv1_noise0/model/', help='saving_path')
# ablation 1
# parser.add_argument('--method', type=str, default='transRWKVb1')
# parser.add_argument('--resuming_model_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_03_21_57_29StransRWKVb1/model/model_341.pth', help='saving_path')
# parser.add_argument('--result_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_03_21_57_29StransRWKVb1/model/', help='saving_path')
# ablation 2
# parser.add_argument('--method', type=str, default='transRWKVb1abalationrwkv')
# parser.add_argument('--resuming_model_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_15_20_10_23StransRWKVb1abalationrwkv/model/model_341.pth', help='saving_path')
# parser.add_argument('--result_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_15_20_10_23StransRWKVb1abalationrwkv/model/', help='saving_path')
# ablation 3
parser.add_argument('--method', type=str, default='transRWKVb1dctv1')
parser.add_argument('--resuming_model_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_20_10_13_32StransRWKVb1dctv1/model/model_343.pth', help='saving_path')
parser.add_argument('--result_path', type=str, default='/home/liumengzu/Code/Demosaic/Compare_Method/Lightprompt/OPTDIC/exp/2025_05_20_10_13_32StransRWKVb1dctv1/model/', help='saving_path')

parser.add_argument("--resume", type=bool, default=True)
parser.add_argument("--noise_level", type=int, default=50)
parser.add_argument('--test_set', type=str, default='NTIRE') # MCAN_REAL KAIST ICVL NTIRE
# Path parameters /home/liumengzu/Data/Spectral/MSFA/KAIST256_16
parser.add_argument('--train_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/NTIRE/ARAD_training/')
parser.add_argument('--valid_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/NTIRE/ARAD_testing/')
# parser.add_argument('--test_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/MCAN_REAL/')
# parser.add_argument('--test_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/KAIST256_16/')
# parser.add_argument('--test_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/ICVL_16/test/')
parser.add_argument('--test_image_dir', type=str, default='/home/liumengzu/Data/Spectral/MSFA/NTIRE/ARAD_testing/test_spectral_16/')
parser.add_argument('--msfa_path', type=str, default='/home/liumengzu/Data/Spectral/MSFA/NTIRE/MSFA_16.mat')

parser.add_argument('--tile', type=int, default=512)
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--tile_overlap', type=int, default=20)

args = parser.parse_args()

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.device == 'cuda':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # torch.cuda.manual_seed(args.seed)

_test(args)



