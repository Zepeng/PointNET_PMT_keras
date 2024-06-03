import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class STNkd(nn.Module):
    '''
        T-Net (k-dimensions)
        Here ONLY for reference but not used due to its negligible performance improvement.
    '''
    def __init__(self, channel=None, k=64, dim_reduce_factor=1):
        super(STNkd, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(channel if channel is not None else k, int(64//dim_reduce_factor), 1)
        self.conv2 = torch.nn.Conv1d(int(64//dim_reduce_factor), int(128//dim_reduce_factor), 1)
        self.conv3 = torch.nn.Conv1d(int(128//dim_reduce_factor), int(1024//dim_reduce_factor), 1)
        self.fc1 = nn.Linear(int(1024//dim_reduce_factor), int(512//dim_reduce_factor))
        self.fc2 = nn.Linear(int(512//dim_reduce_factor), int(256//dim_reduce_factor))
        self.fc3 = nn.Linear(int(256//dim_reduce_factor), k * k)
        self.relu = nn.ReLU()
        self.dim_reduce_factor = dim_reduce_factor

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0] ## Q. why [0]? ## max(keepdim=True) returns (tensor, indices)
        x = x.view(-1, int(1024//self.dim_reduce_factor))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        ## add identity matrix for numerical stability. 
        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
    
class PointNetfeat(nn.Module):
    '''
    PointNet (Encoder) Implementation
    1. STNKD (DISCARDED due to no improvement)
    2. 1x1 Conv1D layers (Literally a linear layer)
    3. Global Statistics (Mean shows superior performance than Min/Max)
    '''
    def __init__(self, dimensions, dim_reduce_factor, args):
        super(PointNetfeat, self).__init__()
        dr = args.enc_dropout

        self.conv1 = torch.nn.Conv1d(dimensions, 64, 1)  # lose a dimension after coordinate transform
        self.conv2 = torch.nn.Conv1d(self.conv1.out_channels, int(128 / dim_reduce_factor), 1)
        self.conv3 = torch.nn.Conv1d(self.conv2.out_channels, int(1024 / dim_reduce_factor), 1)
        self.dr1 = nn.Dropout(dr)
        self.dr2 = nn.Dropout(dr)

        ## recording output dim for construct decoder's input dim
        self.latent_dim = self.conv3.out_channels

    def stats(self, x):
        meani = torch.mean(x, 2, keepdim=True)
        return meani
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.dr1(x))
        x = F.relu(self.dr2(self.conv2(x)))
        x = self.conv3(x)

        global_stats = self.stats(x)
        x = global_stats.squeeze(-1)
        return x
    
class PointClassifier(torch.nn.Module):
    def __init__(self, n_hits, dim, dim_reduce_factor, out_dim, args):
        '''
        Main Model
        :param n_hits: number of points per point cloud
        :param dim: total dimensions of data (3 spatial + time and/or charge)
        '''
        super(PointClassifier, self).__init__()
        dr = args.dec_dropout
        self.n_hits = n_hits
        self.encoder = PointNetfeat(dimensions=dim-1,
                                    dim_reduce_factor=dim_reduce_factor,
                                    args=args,
                                    )
        self.latent = self.encoder.latent_dim ## dimension from enc to dec
        self.decoder = nn.Sequential(
            nn.Linear(self.latent,int(512/dim_reduce_factor)),
            nn.LeakyReLU(),
            nn.Dropout(p=dr),
            nn.Linear(int(512/dim_reduce_factor),int(128/dim_reduce_factor)),
            nn.LeakyReLU(),
            nn.Linear(int(128/dim_reduce_factor),out_dim)
        )

    def process_data_with_label(self,x,label):
        '''
            zero out xyz positions of sensors that are not activated 
        '''
        ## add dimension to allow broadcasting
        t = x[:,4,:].view(-1, 1, self.n_hits)
        q = x[:,5,:].view(-1, 1, self.n_hits)
        chamfer_x = torch.cat([x[:,:3,:], t, q], dim=1)
        label = label.view(-1, 1, self.n_hits).expand(chamfer_x.size())
        ## zero out sensors not activated (including the position features as well)
        x = chamfer_x * label
        return x

    def forward(self, x):
        x = self.process_data_with_label(x, x[:,3,:]) ## Output: [B, F-1, N]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

