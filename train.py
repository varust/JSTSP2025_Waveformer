# coding:utf-8


import torch
import torch.utils.data as tud
import os
import time
import datetime
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from dataset_real import Dataset
from utils import prepare_NTIRE_data, find_last_checkpoint, print2txt, rearrange_channel, save_image, compare_psnr,torch_psnr,torch_ssim,compute_psnr
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as sio
def adjust_learning_rate(optimizer, epoch,args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""

    lr = args.lr * (0.5 ** (epoch // args.step))
    return lr

def _train(model, args, logger):
    #raw,gt, num_dataset = prepare_NTIRE_data(args)
    psnr_max = 0
    #print(model.module.WB_Conv.requires_grad)
    for param in model.module.WB_Conv.parameters():
        param.requires_grad = False
    #trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    #optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch, eta_min=args.eta_min)
    #optim_params = model.parameters()

    # scheduler = MultiStepLR(optimizer,
    #                         milestones=[500,1000,1500,2000,2500,3000,3500,4000], gamma=0.5)
    criterion = torch.nn.L1Loss()
    if args.resume:
        models1_pretrain = torch.load(os.path.join(args.resuming_model_path), map_location='cpu')  # .cuda()
        model.load_state_dict(models1_pretrain['model'],strict=False)
    start_epoch = 0
    '''start_epoch = find_last_checkpoint(model_dir=args.model_path)  # load the last model in matconvnet style
    if start_epoch > 0:
        print('Load model: resuming by loading epoch %03d' % start_epoch)
        checkpoint = torch.load(os.path.join(args.model_dir, 'model_%03d.pth' % start_epoch), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1'''

    iteration = args.num_trainset // args.batch_size

    for epoch_idx in range(start_epoch, args.max_epoch):
        if epoch_idx >= args.resume_epoch:
            model.train()
            args.mode = 'train'

            # lr = adjust_learning_rate(optimizer, epoch_idx+1 ,args)
            # logger.info(f"Learning Rate: {lr}")
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = lr

            train_dataset = Dataset(args)
            train_dataloader = tud.DataLoader(train_dataset, num_workers=args.num_worker, batch_size=args.batch_size, shuffle=True)

            epoch_loss = 0
            start_time = time.time()
            #print(model.module.WB_Conv.weight.data)
            #print(model.module.WB_Conv.requires_grad)
            '''for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    logger.info(param.keys())'''
            for iter_idx, (input, label, mask) in enumerate(train_dataloader):
                input, label, mask = input.to(args.device), label.to(args.device), mask.to(args.device)

                pred = model(input,mask)

                optimizer.zero_grad()
                loss = criterion(pred, label).mean()
                epoch_loss += loss.item()
                # this attribute is added by timm on one optimizer (adahessian)
                loss.backward()
                optimizer.step()

                if iter_idx % (iteration // 4) == 0:
                    logger.info('%4d %4d / %4d loss = %.10f time = %s' % (
                        epoch_idx + 1, iter_idx, iteration, epoch_loss / ((iter_idx + 1) * args.batch_size), datetime.datetime.now()))


            scheduler.step()

            logger.info(f'epoch = {epoch_idx + 1}   lr = {optimizer.param_groups[0]["lr"]} ')


            elapsed_time = time.time() - start_time

            logger.info('epcoh = %4d , loss  = %.10f , time = %4.2f s' % (epoch_idx + 1, epoch_loss / len(train_dataset), elapsed_time))
        else:
            scheduler.step()
            #lr = adjust_learning_rate(optimizer, epoch_idx+1 ,args)
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        if epoch_idx >= args.resume_epoch:

            args.mode = 'test'
            model.eval()
            test_dataset = Dataset(args)
            test_dataloader = tud.DataLoader(test_dataset, batch_size=1)

            psnr_total = 0
            k = 0
            recon_time_total = 0

            with torch.no_grad():
                # Hardware warm-up
                # main test
                psnr_list,psnr_list_v, ssim_list = [], [], []
                pred_list, truth_list = [], []
                begin = time.time()
                for iter_idx, (input, label, mask) in enumerate(test_dataloader):
                    input,label, mask = input.to(args.device),label.to(args.device), mask.to(args.device)

                    tm = time.time()
                    pred = model(input, mask)
                    recon_time = time.time() - tm

                    result = pred
                    result[result < 0.] = 0.
                    result[result > 1.] = 1.

                    for k in range(label.shape[0]):
                        #psnr_val_v = compute_psnr(result[k, :, :, :].detach().cpu().numpy(), label[k, :, :, :].detach().cpu().numpy(), 1)
                        psnr_val = torch_psnr(result[k, :, :, :], label[k, :, :, :])
                        ssim_val = torch_ssim(result[k, :, :, :], label[k, :, :, :])

                        # psnr_list_v.append(psnr_val_v)
                        psnr_list.append(psnr_val.detach().cpu().numpy())
                        ssim_list.append(ssim_val.detach().cpu().numpy())
                        pred_list.append(pred.detach().cpu().numpy())
                        truth_list.append(label.detach().cpu().numpy())
                end = time.time()

                psnr_mean_v = np.mean(np.asarray(psnr_list))
                # psnr_mean = np.mean(np.asarray(psnr_list))
                ssim_mean = np.mean(np.asarray(ssim_list))
                # print(recon_time)

            logger.info('model %d, Avg PSNR = %.4f, Avg SSIM = %.4f, Avg time = %.4f' % (epoch_idx + 1, psnr_mean_v, ssim_mean,(end - begin)))

            if psnr_mean_v > psnr_max:
                psnr_max = psnr_mean_v
                if psnr_max > 40:
                    #name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch_idx+1, psnr_max, ssim_mean) + '.mat'
                    name_img = args.result_path + '/best.mat'
                    #sio.savemat(name_img, {'truth': truth_list, 'pred': pred_list, 'psnr_list': psnr_list_v, 'ssim_list': ssim_list})
                    checkpoint_path = os.path.join(args.model_path, 'model_%03d.pth' % (epoch_idx + 1))
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'epoch': epoch_idx, 'args': args
                                }, checkpoint_path)