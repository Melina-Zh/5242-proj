import argparse
import glob
import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from models import Encoder, DecoderWithAttention
from torchvision import transforms, datasets
import torchvision

C, H, W = 3, 256, 256

def extract_feats(params, model):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))

    train_val_frame_path = os.path.join(params['video_path'] + 'train/')
    train_val_frame_path_list = os.listdir(os.path.join(params['video_path'] + 'train'))
    #print("debug train_val_frame_path_list",train_val_frame_path_list)
    test_frame_path_list = os.listdir(os.path.join(params['video_path'] + 'test'))

    data_transform = transforms.Compose([  # 对读取的图片进行以下指定操作
        transforms.Resize((H, W)),         # 图片放缩为 (300, 300), 统一处理的图像最好设置为统一的大小,这样之后的处理也更加方便
        transforms.ToTensor(),             # 向量化,向量化时 每个点的像素值会除以255,整个向量中的元素值都在0-1之间      
    ])
    
    train_val_frame_path_list = sorted(train_val_frame_path_list)
    image_folder = datasets.ImageFolder(train_val_frame_path,transform = data_transform)
    
    for frame_path in tqdm(train_val_frame_path_list, desc='processing train_val video data'):
        video_id = int(frame_path)
        #print("video id",video_id)
        outfile = os.path.join(dir_fc, f'video{video_id}.npy')
        if os.path.exists(outfile):
            continue
        images = torch.zeros((30, C, H, W))
        for i in range(30):
            image = image_folder[video_id*30+i][0]
            images[i] = image
        with torch.no_grad():
            if params['gpu'] != '-1':
                images = images.cuda()
            fc_feats = model(images).squeeze()
        print(len(fc_feats))
        img_feats = fc_feats.cpu().numpy()
        # Save the video features
        np.save(outfile, img_feats)

    for frame_path in tqdm(test_frame_path_list, desc='processing test video data'):
        video_id = int(frame_path) + len(train_val_frame_path_list)
        outfile = os.path.join(dir_fc, f'video{video_id}.npy')
        if os.path.exists(outfile):
            continue

        images = torch.zeros((30, C, H, W))
        for i in range(30):
            image = image_folder[int(frame_path)*30+i][0]
            images[i] = image
        with torch.no_grad():
            if params['gpu'] != '-1':
                images = images.cuda()
            fc_feats = model(images).squeeze()
        print(len(fc_feats))
        img_feats = fc_feats.cpu().numpy()
        # Save the video features
        np.save(outfile, img_feats)


if __name__ == '__main__':

    resnet = torchvision.models.resnet152(pretrained=True)  # pretrained ImageNet ResNet-152
    modules = list(resnet.children())[:-1] # Remove linear and pool layers (since we're not doing classification)
    resnet = nn.Sequential(*modules)
    params = dict()
    params['output_dir'] = "cs-5242-project-nus-2021-semester2/resnet152"
    params['video_path'] = "cs-5242-project-nus-2021-semester2/"
    params['n_frame_steps'] = 40
    params['gpu'] = '-1'
    extract_feats(params,resnet)
