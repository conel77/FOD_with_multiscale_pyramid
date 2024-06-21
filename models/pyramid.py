import os
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Pyramid(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool2d = nn.MaxPool2d(3, stride=1, padding=1, dilation= 1)


    def forward(self, x1, x2):
        upfeats = x1 #(1,1024,512)

        N, _, C = upfeats.shape
        upfeats = upfeats.permute(0, 2, 1).reshape(N, C, 32, 32)
        #print('upfeats', upfeats.shape)

        downfeats = x2 #(1,256,1024)
        #print('output_feat[1] shape before interpolate:', output_feat[1].shape)
        N, _, C = downfeats.shape
        downfeats = downfeats.permute(0, 2, 1).reshape(N, C, 16, 16)
        #print('downfeats', downfeats.shape) #(1,1024,16,16)
        layers = F.interpolate(downfeats, size=(32,32), mode='bilinear', align_corners=False)
        #print('downfeats222', layers.shape) #(1,1024,32,32)
        layers = self.conv2d(layers)
        #layers = F.upsample(output_feat[1], scale_factor= 2, mode='linear')
        layerss = self.softmax(layers)
        ##Masking = max_pool(Nsoftmax) + max_pool(softmax) 
        nsoftmax =  self.maxpool2d(-1*layerss)
        psoftmax =  self.maxpool2d(layerss)
        Mask_out = nsoftmax + psoftmax

        ## Element Wise multiplication
        #print('layer', layers.shape) #(1,512,32,32)
        #print('ss', Mask_out.shape) #(1,512,32,32)
        output4x = torch.mul(layers, Mask_out)

        output4x = upfeats + output4x
        #print('output', output4x.shape) #(1,512,32,32)

        N, C, _, _ = output4x.shape

        output4x = output4x.reshape(N,C,1024).permute(0, 2, 1)
        #print('output222', output4x.shape) #(1,512,32,32)

        return output4x