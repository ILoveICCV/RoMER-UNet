import sys
import os
from loader import get_loader
from models.Net import Net

sys.path.append(os.getcwd())
from utils.loss_function import BceDiceLoss
from utils.tools import continue_train, get_logger, calculate_params_flops,set_seed
import torch
import argparse

TEST,TRAIN = 1,2
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="Monu_Seg",
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
        default=[10,20,30,40],
        help="out_channels",
    )
    return parser.parse_args()

from train_val_epoch import train_epoch,val_epoch

def main():
    
    torch.cuda.set_device(args.gpu)
    print("Current device ID:", torch.cuda.current_device())
    set_seed(42)
    torch.cuda.empty_cache()
    args=parse_args()
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = get_logger('train', os.path.join(os.getcwd(),args.log,args.datasets))
    #Network
    model=Net(out_channels=args.out_channels,args=args)
    model = model.cuda()
    #loss function
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = 0.001,
            betas = (0.9,0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            amsgrad = False
        )
    #set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 50,
            eta_min = 0.00001,
            last_epoch = -1
        )
    start_epoch=0
    #continue to train
    if args.continues:
        model,start_epoch,min_loss,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_loss={min_loss},lr={lr}')
    #training sets
    train_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TRAIN)
    #testing sets
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TEST)
    min_loss=0
    start_epoch=0
    end_epoch=args.epoch
    steps=0
    #start to run the model
    for epoch in range(start_epoch, end_epoch):
        torch.cuda.empty_cache()
        #train model
        steps=train_epoch(train_loader,model,criterion,optimizer,scheduler,epoch, steps,logger,save_cycles=20)
        # validate model
        loss,miou=val_epoch(val_loader,model,criterion,logger)
        if miou>min_loss:
            print('save best.pth')
            min_loss=miou
            min_epoch=epoch
            torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(checkpoint_path, 'best.pth'))




if __name__ == '__main__':
    main()
