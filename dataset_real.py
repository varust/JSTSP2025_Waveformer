# coding:utf-8

from os import listdir
from os.path import join
import torch
import torch.utils.data as tud
import numpy as np
import random
import scipy.io as sio
from utils import gen_mask, gen_measurement, rearrange_channel
import hdf5storage
from libtiff import TIFFfile
def load_raw(filepath):
    mat = hdf5storage.loadmat(filepath)
    img = mat['mosaic']
    return img

def load_target(filepath):
    mat = hdf5storage.loadmat(filepath)
    # ARAD Dataset
    img = mat['cube']
    norm_factor = mat['norm_factor']

    # Chikusei Dataset
    # data = mat['crop_gt']
    # img = data[0,0]['cube']
    # norm_factor = data[0,0]['norm_factor']
    return img, norm_factor
def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    # tif = TIFFfile(filepath+'/IMECMine_D65.tif')
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0]#.transpose(2, 1, 0)
    # img_test = Image.fromarray(img[:,:,1])

    return img
def rand_crop(target,mask, crop_size):
    [h, w, _] = target.shape
    Height = random.randint(0, (h - crop_size[0]))
    Width = random.randint(0, (w - crop_size[1]))
    return target[Height:(Height + crop_size[0]), Width:(Width + crop_size[1]), :], mask[Height:(Height + crop_size[0]), Width:(Width + crop_size[1]),:]
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
def reorder_imecMCAN(old):
    ### reorder the multiband cube, making the center wavelength from small to large
    _,_,C = old.shape
    new = np.zeros_like(old)
    if C == 16:
        new[:, :, 0] = old[:, :, 2]
        new[:, :, 1] = old[:, :, 0]
        new[:, :, 2] = old[:, :, 9]
        new[:, :, 3] = old[:, :, 1]
        new[:, :, 4] = old[:, :, 15]
        new[:, :, 5] = old[:, :, 14]
        new[:, :, 6] = old[:, :, 12]
        new[:, :, 7] = old[:, :, 13]
        new[:, :, 8] = old[:, :, 7]
        new[:, :, 9] = old[:, :, 6]
        new[:, :, 10] = old[:, :, 4]
        new[:, :, 11] = old[:, :, 5]
        new[:, :, 12] = old[:, :, 11]
        new[:, :, 13] = old[:, :, 10]
        new[:, :, 14] = old[:, :, 8]
        new[:, :, 15] = old[:, :, 3]
        return new
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

def random_channel_shuffle(image):
    # 获取图像的形状
    H, W, C = image.shape

    # 生成通道维度的随机索引
    channel_indices = np.arange(C)
    np.random.shuffle(channel_indices)

    # 应用随机索引对通道进行重排
    shuffled_image = image[:, :, channel_indices]

    return shuffled_image
class Dataset(tud.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()

        self.noise_level = args.noise_level
        self.crop_size = args.patch_size

        #self.num_dataset = num_dataset
        self.mode = args.mode
        if args.dataset == 'NTIRE':
            if args.mode == 'train':
                mosaic_dir = join(args.train_image_dir, "train_mosaic")
                target_dir = join(args.train_image_dir, "train_spectral_16")
                self.augment = True
                self.num = args.num_trainset
            elif args.mode == 'test':
                mosaic_dir = join(args.valid_image_dir, "test_mosaic")
                target_dir = join(args.valid_image_dir, "test_spectral_16")
                self.augment = False
                self.num = 50
            self.image_filenames = [x.split('.')[0] for x in listdir(mosaic_dir)]
            self.target_files = [join(target_dir, fn+"_16.mat") for fn in self.image_filenames]
            self.target_files.sort()
        elif args.dataset == 'CAVE':
            if args.mode == 'train':
                image_dir = join(args.train_image_dir)
                self.augment = True
                self.num = args.num_trainset
                self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir))]
                self.image_filenames.sort()
            elif args.mode == 'test':
                image_dir = join(args.valid_image_dir)
                self.augment = False

                self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir))]
                self.image_filenames.sort()
                self.num = len(self.image_filenames)
        self.num_channel = args.num_channel
        # msfa = sio.loadmat(args.msfa_path)['msfa']
        self.mask = gen_mask(torch.from_numpy(sio.loadmat(args.msfa_path)['msfa']).permute(2, 0, 1), (self.num_channel, args.test_size[0], args.test_size[1]), args.device)
        self.mask = self.mask.transpose(1, 2, 0)
        self.dataset = args.dataset
    def __getitem__(self, index):
        mask = self.mask
        #print(self.mosaic_files[index])
        if self.dataset == 'NTIRE':
            if self.mode == 'train':
                index = random.randint(0, len(self.target_files)-1)
            elif self.mode == 'test':
                index = index  # random.randint(0, 900)
            # raw = load_raw(self.mosaic_files[index])
            # raw = raw.astype(np.float32)
            target, _ = load_target(self.target_files[index])
            target = target.astype(np.float32)
        elif self.dataset == 'CAVE':
            if self.mode == 'train':
                index = random.randint(0, len(self.image_filenames)-1)
            elif self.mode == 'test':
                index = index  # random.randint(0, 900)

            target = load_img(self.image_filenames[index])
            target = target.astype(np.float32) / 255. # 16 512 512
            target = np.transpose(target, (1, 2, 0))


        if self.augment:
            if np.random.uniform() < 0.5:
                target = np.fliplr(target)
                mask = np.fliplr(self.mask)
            if np.random.uniform() < 0.5:
                target = np.flipud(target)
                mask = np.flipud(self.mask)
            k = random.randint(1, 4)
            target = np.rot90(target, k)
            mask = np.rot90(mask, k)
            target, mask = rand_crop(target, mask, self.crop_size)
        # Create RAW Mosaic from Cube
        if self.dataset == 'NTIRE':
            target = reorder_imecNtire(target)# NTIRE -> small2large
            target = reorder_2filter(target)# small2large -> MCANreal_mosaic
        #mask = reorder_imecMCAN(mask)
        target = random_channel_shuffle(target) # channel shuffle
        input_image = mask_input(target, 4)# genereate real mosaic
        #target = random_channel_shuffle(target)
        #input_image = reorder_imecMCAN(input_image)
        raw = input_image.sum(axis=2)
        #target = reorder_imecMCAN(target)

        raw = np.expand_dims(raw, axis=0)
        target = np.transpose(target, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask = mask.astype(np.float32)
        target = target.astype(np.float32)

        raw_input = torch.from_numpy(raw)
        raw_input = raw_input + torch.randn_like(raw_input)*self.noise_level/255
        mask = torch.from_numpy(mask)
        label = torch.from_numpy(target)
        #print(raw_input.shape,label.shape,mask.shape)
        return raw_input, label, mask

    def __len__(self):
        # if self.dataset == 'NTIRE':
        #     return len(self.mosaic_files)
        # elif self.dataset == 'CAVE':
        #     return len(self.image_filenames)
        return self.num#len(self.mosaic_files)
