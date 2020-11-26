import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pdb
def normalize(A , symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)


class GCN(nn.Module):
    def __init__(self , A, dim_in , dim_out):
        super(GCN,self).__init__()
        self.fc1 = nn.Linear(dim_in ,dim_in,bias=True)
        self.fc2 = nn.Linear(dim_in,dim_in//2,bias=True)
        self.fc3 = nn.Linear(dim_in//2,dim_out,bias=True)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.A = normalize(torch.from_numpy(A),True).cuda()

    def forward(self,X):
        '''
        计算三层gcn
        '''
        #建立相似度矩阵
        X = X.reshape((-1,self.dim_in))
        X = F.relu(self.fc1(self.A.mm(X)))
        X = F.relu(self.fc2(self.A.mm(X)))
        X = self.fc3(self.A.mm(X))
        kernel_size = X.size(0)
        X = X.T
        X = X.unsqueeze(dim = 0)
        X = F.avg_pool1d(X, kernel_size)
        return X



if __name__=='__main__':
    #获得空手道俱乐部数据
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).todense()
    #A需要正规化
    A_normed = normalize(torch.FloatTensor(A),True)

    N = len(A)
    X_dim = 20

    # 没有节点的特征，简单用一个单位矩阵表示所有节点
    X = torch.eye(N,X_dim)
    # 正确结果
    Y = torch.zeros(N,1).long()
    # 计算loss的时候要去掉没有标记的样本
    Y_mask = torch.zeros(N,1,dtype=torch.bool)
    # 一个分类给一个样本
    Y[0][0]=0
    Y[N-1][0]=1
    #有样本的地方设置为1
    Y_mask[0][0]=1
    Y_mask[N-1][0]=1

    #真实的空手道俱乐部的分类数据
    Real = torch.zeros(34 , dtype=torch.long)
    for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22] :
        Real[i-1] = 0
    for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34] :
        Real[i-1] = 1

    # 我们的GCN模型
    gcn = GCN(A_normed ,X_dim,2)
    #选择adam优化器
    gd = torch.optim.Adam(gcn.parameters())
    
    for i in range(300):
        #转换到概率空间
        y_pred =F.softmax(gcn(X),dim=1)
        #下面两行计算cross entropy
        loss = (-y_pred.log().gather(1,Y.view(-1,1)))
        #仅保留有标记的样本
        loss = loss.masked_select(Y_mask).mean()

        #梯度下降
        #清空前面的导数缓存
        gd.zero_grad()
        #求导
        loss.backward()
        #一步更新
        gd.step()

        if i%20==0 :
            _,mi = y_pred.max(1)
            #print(mi)
            #计算精确度
            print((mi == Real).float().mean())