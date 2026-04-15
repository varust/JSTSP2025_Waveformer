# coding:utf-8
import torch.nn.functional as F
import torch
import os
import numpy as np
import random
import scipy.io as sio
import glob
import re
from libtiff import TIFFfile
from sklearn.metrics import mean_squared_error
import math
import hdf5storage
from math import exp
from torch.autograd import Variable
import logging

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



def compute_mse(a, b):
    """
    Compute the mean squared error between two arrays

    :param a: first array
    :param b: second array with the same shape

    :return: MSE(a, b)
    """
    assert a.shape == b.shape
    diff = a - b
    return np.power(diff, 2)


def compute_rmse(a, b):
    """
    Compute the root mean squared error between two arrays

    :param a: first array
    :param b: second array with the same shape

    :return: RMSE(a, b)
    """
    sqrd_error = compute_mse(a, b)

    return np.sqrt(np.mean(sqrd_error))


def compute_psnr(a, b, peak):
    """
    compute the peak SNR between two arrays

    :param a: first array
    :param b: second array with the same shape
    :param peak: scalar of peak signal value (e.g. 255, 1023)

    :return: psnr (scalar)
    """
    sqrd_error = compute_mse(a, b)
    mse = sqrd_error.mean()
    # TODO do we want to take psnr of every pixel first and then mean?
    return 10 * np.log10((peak ** 2) / mse)
"""*******************************ssim_compute********************************************"""

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

"""*******************************************************************************"""
def torch_psnr(img, ref):  # input
    img = (img*256).round()
    ref = (ref*256).round()
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input ch h w

    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))
def check_dir(dir_list):
    for i in range(len(dir_list)):
        if not os.path.exists(dir_list[i]):
            os.makedirs(dir_list[i])


def print2txt(txt_path, info):
    with open(txt_path, 'a', encoding='utf-8') as file:
        file.write(info + '\n')


def data_parallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    #gpu_list = list(range(gpu0, gpu0+ngpus))
    #print(ngpus,gpu_list)
    #assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.to('cuda')
    elif ngpus == 1:
        model = model.to('cuda')
    return model


def get_data_dir(data_txt_path, mode='test', data='MSFA'):
    with open(data_txt_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip('\n')
            if data.lower() in line.lower() and mode.lower() in line.lower():
                return line


def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)


def load_path(data_path, mode):
    pathlist = [os.path.join(data_path, name) for name in os.listdir(data_path)]  # 获取data_path路径下所有文件的路径
    if mode == 'train':
        random.shuffle(pathlist)
    else:
        pathlist.sort()
    return pathlist


def load_img(filepath):
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].astype(np.float64)
    return img


def prepare_data(args):
    files_path = load_path(args.data_dir, args.mode)
    file_num = len(files_path)

    msi = []

    for idx in range(file_num):
        # read HrHSI
        data = load_img(files_path[idx])  # c, h, w
        data /= 255.
        msi.append(data)

    msi = np.array(msi).transpose(1, 2, 3, 0)  # num, c, h, w
    msi[msi < 0.] = 0.
    msi[msi > 1.] = 1.
    print(msi.shape)
    return msi, msi.shape[-1]
def load_raw(filepath):
    mat = hdf5storage.loadmat(filepath)
    img = mat['mosaic']
    return img

def load_target(filepath):
    mat = hdf5storage.loadmat(filepath)
    # ARAD Dataset
    img = mat['cube']
    norm_factor = mat['norm_factor']
    return img
def prepare_NTIRE_data(args):

    path_raw_list = [os.path.join(args.NTIRE_train_raw_dir, name) for name in os.listdir(args.NTIRE_train_raw_dir)]  # 获取data_path路径下所有文件的路径
    file_num = len(path_raw_list)
    gt_list = []
    raw_list = []
    for idx in range(file_num):
        # read HrHSI
        data = load_raw(path_raw_list[idx])  # c, h, w
        data = np.expand_dims(data,0)
        raw_list.append(data)

    raw = np.array(raw_list).transpose(1, 2, 3, 0)  # num, c, h, w
    print(raw.shape)
    path_gt_list = [os.path.join(args.NTIRE_train_gt_dir, name) for name in os.listdir(args.NTIRE_train_gt_dir)]  # 获取data_path路径下所有文件的路径
    file_num = len(path_gt_list)
    for idx in range(file_num):
        # read HrHSI
        data = load_target(path_gt_list[idx])  # c, h, w
        gt_list.append(data)

    gt = np.array(gt_list).transpose(1, 2, 3, 0)  # num, c, h, w
    # msi[msi < 0.] = 0.
    # msi[msi > 1.] = 1.
    print(gt.shape)
    return raw,gt, gt.shape[-1]


def find_last_checkpoint(model_dir):
    file_list = glob.glob(os.path.join(model_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def gen_mask(msfa, size, device):
    _, w, h = size
    p = msfa.size(1)
    quotient_w, remainder_w = divmod(w, p)
    quotient_h, remainder_h = divmod(h, p)
    if remainder_w == 0 and remainder_h == 0:
        mask = msfa.repeat(1, quotient_w, quotient_h)
    else:
        mask = torch.zeros(size=size)#.to(device)
        mask[:, :(quotient_w * p), :(quotient_h * p)] = msfa.repeat(1, quotient_w, quotient_h)
        if remainder_w > 0:
            mask[:, -remainder_w:, :] = mask[:, 0:remainder_w, :]
        if remainder_h > 0:
            mask[:, :, -remainder_h:] = mask[:, :, 0:remainder_h]
    return mask.numpy()


def gen_measurement(label, mask):
    return torch.sum(label.mul(mask), dim=0, keepdim=True)


def rearrange_channel(in_spectral, device='cpu'):
    index = [2, 0, 9, 1, 15, 14, 12, 13, 7, 6, 4, 5, 11, 10, 8, 3]
    out_spectral = torch.zeros_like(in_spectral).to(device)
    for i in range(len(index)):
        out_spectral[i, :, :] = in_spectral[index[i], :, :]
    return out_spectral


def save_model(model_dir, epoch_idx, model):
    path_checkpoint = os.path.join(model_dir, 'model_%03d.pth' % (epoch_idx + 1))
    torch.save(model, path_checkpoint)


def save_image(image, image_dir, image_name, num):
    image = image.cpu().permute(2, 3, 1, 0).squeeze(3).numpy()
    save_path = os.path.join(image_dir, image_name + '_%03d.mat' % (num + 1))
    sio.savemat(save_path, {image_name: image})


def compare_psnr(x_true, x_pred):
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true = x_true[:,:,:]
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pred_k = x_pred[:, :, k].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )

        MAX_k = np.max(x_true_k)
        if MAX_k != 0 :
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            # print ('P', PSNR[k])
        else:
            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()
    # mse = MSE.mean()
    # print('psnr', psnr)
    # print('mse', mse)
    return psnr


if __name__ == '__main__':
    pass
