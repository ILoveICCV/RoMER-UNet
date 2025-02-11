import random
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

TEST,TRAIN = 1,2
class ISIC2018_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        gts_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1_Training_GroundTruth','ISIC2018_Task1_Training_GroundTruth')
        images_path=os.path.join(cwd,'data','ISIC2018','ISIC2018_Task1-2_Training_Input','ISIC2018_Task1-2_Training_Input')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        
        if mode==TRAIN:
            self.data=self.data[:2076]
        elif mode==TEST:
            self.data=self.data[2076:2594]
        print(len(self.data))
        self.data_buf=self.cuda_buffer()
        
    #put datasets into inner memory, improving the training speed.
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name     

    def __len__(self):
        return len(self.data)


class PH2_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd=os.getcwd()
        images_path=os.path.join(cwd,'data','PH2','PH2Dataset','PH2 Dataset images')
        images_list=sorted(os.listdir(images_path))
        random.shuffle(images_list)
        self.data=[]
        for path in images_list:
            image_path=os.path.join(images_path,path,path+'_Dermoscopic_Image',path+'.bmp')
            gt_path=os.path.join(images_path,path,path+'_lesion',path+'_lesion.bmp')
            self.data.append([image_path, gt_path])
     
        self.data=self.data[0:2]
        self.transformer=transformer
        print(f'the length of datasets is {len(self.data)}')
    
    def __getitem__(self, index):
        image_path, gt_path=self.data[index]
        image = Image.open(image_path).convert('RGB')
        image=np.array(image)
        image = np.transpose(image, axes=(2, 0, 1))
        gt = Image.open(gt_path).convert('L')
        gt = np.array(gt)
        gt=np.expand_dims(gt, axis=2) / 255
        gt = np.transpose(gt, axes=(2, 0, 1))
        image, gt = self.transformer((image, gt))
        if self.mode==TEST:
            # return image,gt,image_path.split('/')[-1]
            return image,gt,image_path.split('\\')[-1]
        return image,gt

    def __len__(self):
        return len(self.data)

class BUSI_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        self.mode=mode
        cwd='/home/xyq/Datasets'
        data_path_1=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','benign')
        data_path_2=os.path.join(cwd,'data','BUSI','Dataset_BUSI','Dataset_BUSI_with_GT','malignant')
        benign_list=sorted(os.listdir(data_path_1))
        malignant_list=sorted(os.listdir(data_path_2))

        benign_image_list=[item for item in benign_list if ").png" in item]
        benign_gt_list=[item for item in benign_list if "mask.png" in item]

        malignant_image_list=[item for item in malignant_list if ").png" in item]
        malignant_gt_list=[item for item in malignant_list if "mask.png" in item]

        
        self.data=[]
        for i in range(len(benign_image_list)):
            image_path=data_path_1+'/'+benign_image_list[i]
            mask_path=data_path_1+'/'+benign_gt_list[i]
            self.data.append([image_path, mask_path])
        for i in range(len(malignant_image_list)):
            image_path=data_path_2+'/'+malignant_image_list[i]
            mask_path=data_path_2+'/'+malignant_gt_list[i]
            self.data.append([image_path, mask_path])

        # random.shuffle(self.data)

        if mode==TRAIN:
            self.data=self.data[:518]
        if mode==TEST:
            self.data=self.data[518:647]
        
        self.transformer=transformer
        print(len(self.data))
        self.data_buf=self.cuda_buffer()




    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf

    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name     

    def __len__(self):
        return len(self.data)




class Kvasir_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        # cwd='/home/xyq/Datasets'
        cwd=os.getcwd()
        self.mode=mode
        gts_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','masks')
        images_path=os.path.join(cwd,'data','Kvasir','kvasir-seg','Kvasir-SEG','images')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "jpg" in item]
        self.data=[]
        for i in range(len(images_list)):
            image_path=images_path+'/'+images_list[i]
            mask_path=gts_path+'/'+gts_list[i]
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        if mode==TRAIN:
            self.data=self.data[:880]
        elif mode==TEST:
            self.data=self.data[880:1000]

        print(len(self.data))
        self.data_buf=self.cuda_buffer()
    

    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('/')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name

    def __len__(self):
        return len(self.data)



class COVID_19_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        gts_path=os.path.join(cwd,'data','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','masks')
        images_path=os.path.join(cwd,'data','COVID_19','COVID-19_Lung_Infection_train','COVID-19_Lung_Infection_train','images')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "jpg" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)

        if mode==TRAIN:
            self.data=self.data[:716]
        elif mode==TEST:
            self.data=self.data[716:894]

        print(len(self.data))
        self.data_buf=self.cuda_buffer()
        
    #put datasets into inner memory, improving the training speed.
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('\\')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name 

    def __len__(self):
        return len(self.data)



class CVC_ClinkDB_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode
        gts_path=os.path.join(cwd,'data','CVC_ClinkDB','archive','PNG','Ground Truth')
        images_path=os.path.join(cwd,'data','CVC_ClinkDB','archive','PNG','Original')
        
        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]

        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        print(len(self.data))
        self.data_buf=self.cuda_buffer()
        
    #put datasets into inner memory, improving the training speed.
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            image = Image.open(image_path).convert('RGB')
            image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('\\')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name 

    def __len__(self):
        return len(self.data)


import tifffile as tiff
class Monu_Seg_Datasets(Dataset):
    def __init__(self,mode,transformer):
        super().__init__()
        cwd=os.getcwd()
        self.mode=mode

        gts_path=os.path.join(cwd,'data','Monu_Seg','archive','kmms_test','kmms_test','masks')
        images_path=os.path.join(cwd,'data','Monu_Seg','archive','kmms_test','kmms_test','images')
        gts_path_=os.path.join(cwd,'data','Monu_Seg','archive','kmms_training','kmms_training','masks')
        images_path_=os.path.join(cwd,'data','Monu_Seg','archive','kmms_training','kmms_training','images')

        images_list=sorted(os.listdir(images_path))
        images_list = [item for item in images_list if "png" in item]
        gts_list=sorted(os.listdir(gts_path))
        gts_list = [item for item in gts_list if "png" in item]


        images_list_=sorted(os.listdir(images_path_))
        images_list_ = [item for item in images_list_ if "tif" in item]
        gts_list_=sorted(os.listdir(gts_path_))
        gts_list_ = [item for item in gts_list_ if "png" in item]

        gts_list = gts_list + gts_list_
        images_list = images_list + images_list_
        
        self.data=[]
        for i in range(len(images_list)):
            image_path=os.path.join(images_path,images_list[i])
            mask_path=os.path.join(gts_path,gts_list[i])
            self.data.append([image_path, mask_path])
        self.transformer=transformer
        random.shuffle(self.data)
        print(len(self.data))

        if mode==TRAIN:
            self.data=self.data[:59]
        elif mode==TEST:
            self.data=self.data[59:74]
        self.data_buf=self.cuda_buffer()
        
    #put datasets into inner memory, improving the training speed.
    def cuda_buffer(self):
        data_buf=[]
        id=0
        for data in self.data:
            image_path,gt_path=data
            if '.tif' in image_path:
                image = tiff.imread('example.tif')
            else:
                image = Image.open(image_path).convert('RGB')
                image=np.array(image)
            image = np.transpose(image, axes=(2, 0, 1))
            gt = Image.open(gt_path).convert('L')
            gt = np.array(gt)
            gt=np.expand_dims(gt, axis=2) / 255
            gt = np.transpose(gt, axes=(2, 0, 1))
            image, gt = self.transformer((image, gt))
            image=image.cuda()
            gt=gt.cuda()
            if self.mode==TEST:
                data_buf.append([image,gt,image_path.split('\\')[-1]])
            else:
                data_buf.append([image,gt])
            if id%20==0:
                print(id)
            id=id+1
        return data_buf
    
    def __getitem__(self, index):
        if self.mode!=TEST:
            image, gt=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt
        else:
            image, gt,image_name=self.data_buf[index]
            image=image.cpu()
            gt=gt.cpu()
            return image,gt,image_name 

    def __len__(self):
        return len(self.data)
