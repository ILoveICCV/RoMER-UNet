import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics


def val_epoch(val_loader,model,criterion,logger):
    '''
    val_loader: val sets
    model: training model
    criterion: loss function
    logger: record log
    '''
    model.eval()
    loss_list=[]
    preds = []
    gts = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            images, gt,image_name = data
            images, gt = images.cuda().float(), gt.cuda().float()
            pred = model(images)
            loss = criterion(pred[0],gt)
            #record val loss
            loss_list.append(loss.item())
            #record gt and pred for the subsequent metric calculation.
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
    #calculate metrics
    log_info,miou=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list),miou


def train_epoch(train_loader,model,criterion,optimizer,scheduler,epoch,steps,logger,save_cycles=5):
    '''
    train_loader: training sets
    model: training model
    criterion: loss function
    optimizer: optimizer
    scheduler: scheduler
    epoch: current epoch
    steps: current step
    logger: record log
    save_cycles: the cycle of printing log
    '''
    model.train()
    loss_list=[]
    for step,data in enumerate(train_loader):
        steps+=step
        optimizer.zero_grad()
        images, gts = data
        images, gts = images.cuda().float(), gts.cuda().float()
        pred=model(images)
        #multi-supervision
        loss=criterion(pred[0],gts)
        for i in range(1,len(pred)):
            loss=loss+criterion(pred[i],gts)
        loss.backward()
        optimizer.step()
        #record training loss
        loss_list.append(loss.item())
        #print log
        if step%save_cycles==0:
            lr=optimizer.state_dict()['param_groups'][0]['lr']
            log_info=f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.7f}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step