
import torchvision 
import pdb
import os
import torchvision.transforms as transforms
from dataset import *
import torch
import torch.nn as nn
from utils import *
import time
import numpy as np
from imageretrievalnet import *
from loss import *
import time
import argparse

from MAP import *


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')


parser.add_argument('--cls-num', default=101, type=int,
                    metavar='N', help='class number')

parser.add_argument('--mul-cls-num', default=174, type=int,
                    metavar='N', help='ingradient class number')
                    
parser.add_argument('--epoch', default=100, type=int,
                    metavar='N', help='epochs')
parser.add_argument('--bs', default=64, type=int,
                    metavar='N', help='batch size')
parser.add_argument('--imsize', default=362, type=int,
                    metavar='N', help='image size')
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='N', help='learning rate')
parser.add_argument('--dataset', default='food101', type=str,
                     help='dataset name')
parser.add_argument('--dataroot', default='/data1/sjj/dataset_food', type=str,
                     help='dataset name')
parser.add_argument('--net', default='resnet101', type=str,
                     help='network')
parser.add_argument('--batch-p', default=8, type=int,
                    metavar='N', help='class per batch')
parser.add_argument('--batch-k', default=4, type=int,
                    metavar='N', help='images per class')

parser.add_argument('--loss', default='cross', type=str,
                    help='loss function')

parser.add_argument('--graph',action='store_true',
                    help='use graph')

parser.add_argument('--test',action='store_true',
                    help='test before training')

parser.add_argument('--cls-only',action='store_true',
                    help='only cross loss')
def main():

    global args
    args = parser.parse_args()
    # model = torchvision.models.resnet101(pretrained=True)
    
    # model.fc = nn.Linear(in_features=2048, out_features=cls_num, bias=True)

    # model = model.cuda()

    

    EPOCHS = args.epoch

    BATCH_SIZE = args.bs
    
    image_size = args.imsize
    
    lr = args.lr

    dataset = args.dataset
 
    root = args.dataroot
    
    ann_folder = os.path.join(root, dataset, 'retrieval_dict')
    
    imgs_root = os.path.join(root, dataset, 'images')

    net_name = args.net

    cls_num = args.cls_num

    mult_cls_num = args.mul_cls_num
    ###############################################
    batch_p = args.batch_p

    batch_k = args.batch_p
     
    
    

    adj = np.load(os.path.join(ann_folder,'adj.npy'))

    meta = {}
    meta['graph'] = args.graph
    meta['adj'] = adj
    meta['outputdim'] = 2048

    model = image_net(net_name,cls_num,mult_cls_num,meta).cuda()

    #model=nn.DataParallel(model,device_ids=[0,1]) 
    ####################################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    

    criterion =  nn.BCEWithLogitsLoss( reduction='mean' )
    
    criterion_cls =  nn.CrossEntropyLoss()

    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    d = localtime+'_'+args.loss;
    
    if args.graph:
        d+='_graph'

    directory = os.path.join(dataset,d)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    ingre_file = os.path.join(ann_folder,'ingre_dict.npy')
    if args.loss == 'cross':
        train_dataset = ImagesForMulCls(imgs_root,os.path.join(ann_folder,'train_full.txt'), ingre_file, image_size,transform=transform)
    elif args.loss == 'triplet':
        train_dataset = TuplesDataset(os.path.join(ann_folder,'train_full.txt'),ingre_file,imgs_root,image_size,batch_p = batch_p,batch_k = batch_k,transform=transform)
        criterion_metric = TripletLoss(batch_p, batch_k, margin=0.85).cuda()
    elif args.loss == 'contrastive':
        train_dataset = TuplesDataset(os.path.join(ann_folder,'train_full.txt'),ingre_file,imgs_root,image_size,batch_p = batch_p,batch_k = batch_k,transform=transform)
        criterion_metric = ContrastiveLoss(batch_p, batch_k, margin=0.85).cuda()
    elif args.loss == 'smoothap':
        train_dataset = TuplesDataset(os.path.join(ann_folder,'train_full.txt'),ingre_file,imgs_root,image_size,batch_p = batch_p,batch_k = batch_k,transform=transform)
        criterion_metric = SmoothAP(anneal, batch_p*batch_k, batch_p, meta['outputdim'] ).cuda() 
    elif args.loss == 'circle':
        train_dataset = TuplesDataset(os.path.join(ann_folder,'train_full.txt'),ingre_file,imgs_root,image_size,batch_p = batch_p,batch_k = batch_k,transform=transform)
        criterion_metric = CircleLoss(batch_p, batch_k, margin=args.loss_margin).cuda()
    else:
        raise (RuntimeError("Loss {} not available!".format(args.loss)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, sampler=None,
    )

    test_dataset = ImagesForMulCls(imgs_root,os.path.join(ann_folder,'test_full.txt'),  ingre_file, image_size,transform=transform)
    if args.graph:
        BATCH_SIZE = 32
    else:
       BATCH_SIZE = 64
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, sampler=None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
    Logger_file = os.path.join(directory,"log.txt")
    
    if args.test:
        AP, precision, mAP, recall = test(test_loader, model, -1)
        print('AP:',AP)
        print('precision:',precision)
        print('mAP:',mAP)
        with open(Logger_file,'a') as f:
            f.write("epoch:{}\tAP@m:{}\tPrecision:{}\tmAP:{}\trecall:{}".format(-1,AP,precision,mAP,recall))
    for epoch in range(EPOCHS):
        
        if args.loss == 'triplet' or 'contrastive' or 'smoothap' or 'circle':
            train_loader.dataset.create_tuple()
            train(train_loader,model,epoch,criterion,criterion_cls,optimizer,args,criterion_metric)
        else:
            train(train_loader,model,epoch,criterion,criterion_cls,optimizer,args)
        
        torch.cuda.empty_cache()
        AP,precision,mAP, recall = test(test_loader, model, epoch)
        print('AP:',AP)
        print('precision:',precision)
        print('mAP:',mAP)
        with open(Logger_file,'a') as f:
            f.write("epoch:{}\tAP@m:{}\tPrecision:{}\tmAP:{}\trecall:{}\n".format(epoch,AP,precision,mAP,recall))
        path = os.path.join(directory,'model_epoch_{}.pth'.format(epoch))
        torch.save(model,path)

def train(train_loader,model,epoch,criterion,criterion_cls, optimizer,args, criterion_metric=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for step, (x, cls, y) in enumerate(train_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.squeeze()
        cls = cls.squeeze()
        y = y.squeeze()
        
        x = x.cuda()
        out_m,out_cls,out = model(x)
        
        

        loss = criterion_cls(out_cls,cls.cuda())#分类损失

        if not args.cls_only:
            multi_loss = criterion(out, y.float().cuda())#多标签损失
            loss = loss+multi_loss

        if criterion_metric != None:
            metric_loss = criterion_metric(out_m)#度量损失
            loss = loss + metric_loss

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 and step != 0:
            print('>> Train: [{0}][{1}/{2}]\tloss:{3:.3f}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(train_loader),loss.item(), batch_time=batch_time,
                data_time=data_time))



def test(test_loader, model, epoch):
    print('>> Evaluating network on test datasets...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.eval()
    ap_meter = AveragePrecisionMeter(False)
    precision = PrecisionMeter(False)
    dataset = []
    clusters = []
    for step, (x, lbl, mul_lbl) in enumerate(test_loader):
        batch_time.update(time.time() - end)
        end = time.time()
        x = x.cuda()
        x = x.contiguous()
        mul_lbl = mul_lbl.cuda()
        
        
        with torch.no_grad():
            vec, out_cls, out = model(x)
        
        precision.add(out_cls.data,lbl)
        ap_meter.add(out.data,mul_lbl)

        dataset.extend(vec.unsqueeze(0))
        clusters.extend(lbl.cpu().numpy())
        
        if step % 100 == 0:
            print('>> Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                epoch+1, step+1, len(test_loader), batch_time=batch_time,
                data_time=data_time))
    
    dataset = torch.cat(dataset, dim = 0)
    mAP,recall = Test(dataset,clusters)
    
    return ap_meter.value().mean(), precision.value(), mAP, recall
if __name__=='__main__':
    main()