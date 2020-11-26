
import os
import pdb

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from GCN import *


class ImageRetrievalNet(nn.Module):
    def __init__(self, features,fc_cls,fc_mul,pool, meta):
        
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.pool = pool
        self.norm = L2N()
        self.graph = meta['graph']
        self.GCN = GCN(meta['adj'],meta['outputdim'],meta['outputdim'])
        if type(fc_cls)==list: 
          self.fc_cls = nn.Sequential(*fc_cls)   
        else:
          self.fc_cls=fc_cls
        if type(fc_mul)==list: 
          self.fc_mul = nn.Sequential(*fc_mul)   
        else:
          self.fc_mul=fc_mul
        self.sigmoid = nn.Sigmoid()
        
        

    

    
    def forward(self, x, test=False):
        o = self.features(x)
        if self.graph:
            params = list(self.fc_mul.parameters())
            weight_softmax = self.sigmoid(params[0])
            weight_softmax = weight_softmax.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            o = o.unsqueeze(1)
            o = o.expand(o.shape[0],weight_softmax.shape[1],o.shape[2],o.shape[3],o.shape[4])
            o = weight_softmax*o

            x = torch.zeros((o.shape[0],o.shape[2],1)).cuda()
            for idx in range(o.shape[0]): 
                x[idx] = self.GCN(self.pool(o[idx]))
            
            o = x.unsqueeze(-1)
        else:
            o = self.pool(o)


            
        cls = self.fc_cls(o.squeeze()) 
        cls_mul = self.fc_mul(o.squeeze()) 

        o = self.norm(o).squeeze(-1).squeeze(-1)
        
        return o, cls, cls_mul


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def l2n(self, x, eps=1e-6):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


    def forward(self, x):
        return self.l2n(x, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


def image_net(net_name,cls_num,mult_cls_num,meta):
    if net_name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif net_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    else:
         raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))
    features = list(net.children())[:-2]
    avg_p = net.avgpool
    fc_cls = nn.Linear(in_features=2048, out_features=cls_num, bias=True)
    fc_mul = nn.Linear(in_features=2048, out_features=mult_cls_num, bias=True)
    pool =  GeM()
    return ImageRetrievalNet(features,fc_cls,fc_mul,pool,meta)