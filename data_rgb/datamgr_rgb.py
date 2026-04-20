# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import data_rgb.additional_transforms as add_transforms
from data_rgb.dataset_rgb import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod


# =====数据预处理=======     只需给定图像尺寸
class TransformLoader:
    def __init__(self, image_size, 
                 # normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 normalize_param=dict(mean=[0.5036, 0.5036, 0.5036], std=[0.1569, 0.1569, 0.1569]),     #### 对应RGB图像
                 # normalize_param=dict(mean=[0.5036, ], std=[0.1569, ]),      #### 对应灰度图像
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':     # 调整图像亮度、对比度、色彩
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)  #获取属性值，即transform_type
        # =======以下为图片尺寸裁剪、归一化等操作
        # if transform_type=='RandomResizedCrop':
        #     return method(self.image_size)
        # elif transform_type=='CenterCrop':
        #     return method(self.image_size)
        # elif transform_type=='Scale':
        if transform_type == 'Resize':
            # return method([int(self.image_size*1.15), int(self.image_size*1.15)])
            return method([int(self.image_size ), int(self.image_size )])
        # elif transform_type=='Normalize':
        #     return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):  #所有数据处理方法整合，包括数据增强和不进行数据增强两种
        if aug:  #数据增强
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            # transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize'] ##原来的
            # transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize'] ##后来改的1
            transform_list = ['Resize',  'ToTensor']  ##后来改的2
            # transform_list = ['Resize', 'Grayscale', 'ToTensor', 'Normalize']
            # transform_list = ['Resize',  'ToTensor']  ##计算均值、偏差！

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager:    #写一个抽象父类
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

#  继承DataManager父类并重写
# ================== 一般数据加载器 ====================
class SimpleDataManager(DataManager):  #输入图像尺寸和批数量
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)    #图像处理

# ==============数据加载器=============
    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)  #数据增强
        dataset = SimpleDataset(data_file, transform)
        # LL = list(dataset)
        # T, L = LL[0]
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

# =================小样本数据加载器 =======================
# class SetDataManager(DataManager):
#     def __init__(self, image_size, n_way, n_support, n_query, n_eposide =10):
#         super(SetDataManager, self).__init__()
#         self.image_size = image_size
#         self.n_way = n_way
#         self.batch_size = n_support + n_query
#         self.n_eposide = n_eposide
#         self.trans_loader = TransformLoader(image_size)
#
#     def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
#         transform = self.trans_loader.get_composed_transform(aug)  #数据增强
#         dataset = SetDataset( data_file , self.batch_size, transform )  #按类别包装
#         sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )     #小样本采样器
#         # data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)
#         data_loader_params = dict(batch_sampler=sampler, num_workers=0, pin_memory=True)  ##----------du
#         data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)  # 包含2层dataloader，第一层是batch，本层是类别
#         return data_loader


