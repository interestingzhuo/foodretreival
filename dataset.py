import numpy as np
import torch
import torch.utils.data as data
import os
import pdb
import cv2
import torchvision.transforms as transforms



class ImagesForMulCls(data.Dataset):
    
    def __init__(self, ims_root, file_list, ingre_file, imsize=224, bbxs=None, transform=None):

        
        self.root = ims_root
        self.images_fn,self.clusters = self.get_imgs(ims_root,file_list)
        #pdb.set_trace()
        self.ingre_dict = np.load(ingre_file)
        self.imsize = imsize
        self.transform = transform

    def get_imgs(self,ims_root,file_list):
        file=open(file_list)
        lines=file.readlines()
        images=[]
        clusters=[]
        for line in lines:
            image=line.split()[0]
            label = line.strip().split()[-1]
            images+=[os.path.join(ims_root, image)]
            #label=[int(l) for l in label]
            clusters+=[int(label)]
        return images,np.array(clusters)
    def __getitem__(self, index):
        img = cv2.imread(self.images_fn[index])
        img = cv2.resize(img,(self.imsize,self.imsize))
        if self.transform is not None:
            img = self.transform(img)

        return img, self.clusters[index], self.ingre_dict[self.clusters[index]]
    def __len__(self):
        return len(self.images_fn)


class TuplesDataset(data.Dataset):

    def __init__(self,train_file,muti_file,imgs_root,imsize,batch_p,batch_k,transform,iterations=2000):
        self.iterations = iterations
        self.train_file = train_file
        self.imsize = imsize
        self.batch_p = batch_p
        self.batch_k = batch_k
        self.transform = transform
        self.imgs_root = imgs_root
        self.dict = self.create_dict(self.train_file,self.imgs_root)
        self.mult_dict = np.load(muti_file)
    
    def create_dict(self,file_list,ims_root):
        file=open(file_list)
        lines=file.readlines()
        dict = {}
        for line in lines:
            image,label=line.split()
            image = os.path.join(ims_root,image)
            label=int(label)
            if label in dict:
                dict[label]+=[image]
            else:
                dict[label] = [image]
        return dict
    def create_tuple(self):
        self.batch_sample = []
        self.batch_label = []
        self.batch_mul_label = []
        for i in range(self.iterations):
            sample_cls = np.random.choice(list(self.dict.keys()),self.batch_p,replace=False)
            imgs = []
            labels = [[cls]*self.batch_k for cls in sample_cls]  
            mul_labels = [[self.mult_dict[cls]]*self.batch_k for cls in sample_cls]
            for key in sample_cls:
                samples_img = np.random.choice(self.dict[key],self.batch_k,replace=False)
                imgs+=[samples_img]
            imgs = np.reshape(imgs,(-1))
            labels = np.reshape(labels,(-1))
            mul_labels = np.reshape(mul_labels,(-1,174))
            self.batch_sample += [imgs]
            self.batch_label += [labels]
            self.batch_mul_label += [mul_labels]
    def __getitem__(self, index):
        
        output = []
        imgs = self.batch_sample[index]
        clss = self.batch_label[index]
        muti_cls = self.batch_mul_label[index]
        for path in imgs:
            
            img = cv2.imread(path)
            output+=[img]
        if self.imsize is not None:
            output = [cv2.resize(img,(self.imsize,self.imsize)) for img in output]

        if self.transform is not None:
            output = [self.transform(o).unsqueeze(0) for o in output]
        output = torch.cat(output,dim = 0)
        return output, clss, muti_cls
    def __len__(self):
        return self.iterations



class dataset(data.Dataset):
    

    def __init__(self, root,transform):

        self.root = root
        self.imsize = 224
        self.transform = transform
        self.images,self.labels = self.load(root)

    def load(self,root):
        clses = os.listdir(root)
        clses = sorted(clses)
        dic = {}
        images = []
        labels = []
        for i in range(len(clses)):
            dic[clses[i]] = i
        for name in clses:
            dir = os.path.join(root,name)
            for img in os.listdir(dir):
                images += [os.path.join(dir,img)]
                labels += [dic[name]]
        #idx = [i for i in range(len(images))]
        #np.random.shuffle(idx)
        #images = [images[i] for i in idx]
        #labels = [labels[i] for i in idx]
        return images,labels

    def __getitem__(self, index):
        
        img = cv2.imread(self.images[index])
        img = cv2.resize(img,(self.imsize,self.imsize))
        lbl = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img,lbl

    def __len__(self):
        return len(self.labels)

if __name__=='__main__':
    img_root = '/data1/sjj/dataset_food/food172/images'
    file_list = '/data1/sjj/dataset_food/food172/retrieval_dict/train_ingredient.txt'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),normalize
    ])
    train_dataset = ImagesForMulCls(img_root,file_list,transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    for step, (x, y) in enumerate(train_loader):
        pdb.set_trace()