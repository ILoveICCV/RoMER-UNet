import sys
import os
from loader import get_loader
from models.Net import RoMER_UNet

sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss
from utils.tools import continue_train, get_logger, calculate_params_flops,set_seed
import torch
import argparse

from mirco import TEST,TRAIN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="Kvasir",
        help="input datasets name including ISIC2018, PH2, Kvasir, BUSI, COVID_19,CVC_ClinkDB,Monu_Seg",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default="8",
        help="input batch_size",
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=256,
        help="input image resolution.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="input log folder: ./log",
    )
    parser.add_argument(
        "--continues",
        type=int,
        default=0,
        help="1: continue to run; 0: don't continue to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoints',
        help="the checkpoint path of last model: ./checkpoints",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu_id:",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=42,
        help="random configure:",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="end epoch",
    )
    parser.add_argument(
        "--out_channels",
        type=list,
        default=[10,20,30,40,50],
        help="out_channels",
    )
    parser.add_argument(
        "--kernel_list",
        type=list,
        default=[3,9],
        help="out_channels",
    )
    parser.add_argument(
        "--save_cycles",
        type=int,
        default=20,
        help="out_channels",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="out_channels",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=50,
        help="out_channels",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="out_channels",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="out_channels",
    )
    return parser.parse_args()

from train_val_epoch import train_epoch,val_epoch

def main():
    #init GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("Current device ID:", torch.cuda.current_device())
    else:
        print("no GPU devices")
    args=parse_args()
    #random
    set_seed(args.random)
    torch.cuda.empty_cache()
    #check folders
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #record log
    logger = get_logger('train', os.path.join(os.getcwd(),args.log,args.datasets))
    #Network
    model=RoMER_UNet(out_channels=args.out_channels,kernel_list=args.kernel_list)
    model = model.cuda()
    #loss function
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.lr,
            betas = (0.9,0.999),
            eps = args.eps,
            weight_decay = args.weight_decay,
            amsgrad = False
        )
    #set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = args.T_max,
            eta_min = 0.00001,
            last_epoch = -1
        )
    start_epoch=0
    min_miou=0
    #continue to train
    if args.continues:
        model,start_epoch,min_miou,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_miou={min_miou},lr={lr}')
    #training sets
    train_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TRAIN)
    #testing sets
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TEST)
    end_epoch=args.epoch
    steps=0
    #start to run the model
    for epoch in range(start_epoch, end_epoch):
        torch.cuda.empty_cache()
        #train model
        steps=train_epoch(train_loader,model,criterion,optimizer,scheduler,epoch, steps,logger,save_cycles=args.save_cycles)
        # validate model
        loss,miou=val_epoch(val_loader,model,criterion,logger)
        #check current mIoU
        if miou>min_miou:
            print('save best.pth')
            min_iou=miou
            torch.save(
            {
                'epoch': epoch,
                'min_miou': min_iou,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, 'best.pth'))
            

if __name__ == '__main__':
    main()
