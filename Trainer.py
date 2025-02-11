
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics

class Trainer():
    def __init__(self,train_loader,val_loader,model,criterion,optimizer,scheduler,start_epoch,end_epoch,logger,logger_curve,checkpoint_path,save_cycles=20,save_path=None):
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.model=model
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.start_epoch=start_epoch
        self.end_epoch=end_epoch
        self.logger=logger
        self.checkpoint_path=checkpoint_path
        self.save_cycles=save_cycles
        self.save_path=save_path
        image_root =  os.path.join(save_path)
        if not os.path.exists(image_root):
            os.makedirs(image_root)
        self.logger_curve=logger_curve

    def run(self):
        #running settings
        min_loss=0
        steps=0
        #start to run the model
        for epoch in range(self.start_epoch, self.end_epoch):
            torch.cuda.empty_cache()
            #train model
            steps=self.train_epoch(epoch,steps)
            # #validate model
            loss,miou=self.val_epoch()
            if miou>min_loss:
                print('save best.pth')
                min_loss=miou
                torch.save(
                {
                    'epoch': epoch,
                    'min_loss': min_loss,
                    'min_epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.checkpoint_path, f'best.pth'))
    

    def val_epoch(self):
        self.model.eval()
        loss_list=[]
        preds = []
        gts = []
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images, gt,image_name = data
                images, gt = images.cuda(non_blocking=True).float(), gt.cuda(non_blocking=True).float()
                pred = self.model(images)
                #计算损失
                loss = self.criterion(pred[0],gt)
                loss_list.append(loss.item())

                
                gts.append(gt.squeeze(1).cpu().detach().numpy())
                preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
        log_info,dsc,miou=get_metrics(preds,gts)
        self.logger_curve[0].info(f'{miou}')
        self.logger_curve[1].info(f'{dsc}')
        self.logger_curve[2].info(f'{np.mean(loss_list)}')
        log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
        print(log_info)
        self.logger.info(log_info)
        return np.mean(loss_list),miou


    def train_epoch(self,epoch,steps):
        self.model.train()
        loss_list=[]
        for step,data in enumerate(self.train_loader):
            steps+=step
            #清空梯度信息
            self.optimizer.zero_grad()
            images, gts = data
            images, gts = images.cuda().float(), gts.cuda().float()
            pred=self.model(images)
            loss=self.criterion(pred[0],gts)
            for i in range(1,len(pred)):
                loss=loss+self.criterion(pred[i],gts)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
            if step%self.save_cycles==0:
                lr=self.optimizer.state_dict()['param_groups'][0]['lr']
                log_info=f'train: epoch={epoch}, step={step}, loss={np.mean(loss_list):.4f}, lr={lr:.7f}'
                print(log_info)
                self.logger.info(log_info)
        self.scheduler.step()
        return step


