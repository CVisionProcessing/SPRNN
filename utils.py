import pandas as pd
import numpy as np
import h5py
import cv2
from torch.utils import data
import torchvision.transforms as transforms
import random
import numpy as np
from torch.utils import data

class ToTensor(object):
    def __call__(self, sample):
        for k in sample:
            sample[k] = torch.from_numpy(np.array(sample[k]).astype(np.float32)).float()
        return sample
    
class SeqDataset(data.Dataset):
    def __init__(self,
                 dataname,data_path,image_path,
                 mode=None,
                 ):
        super().__init__()
        hf = h5py.File(data_path, 'r')
        self.data = np.array(hf['data'])
        if dataname=='TaxiBJ':

            roadgraph = cv2.imread(image_path) 
            ret, thresh1 = cv2.threshold(roadgraph, 1, 1, cv2.THRESH_BINARY)
            thresh1 = cv2.dilate(thresh1,np.ones((3,3),np.uint8))
            thresh1 = cv2.resize(thresh1, (512, 512), interpolation=cv2.INTER_NEAREST)
            self.roadgraph = thresh1[:,:,0]
            self.mean  = self.data[:20000].mean()
            self.var = self.data[:20000].std()
            self.data = (self.data-self.mean)/self.var
            d_shift = 8
            self.long_seq_shift = [48*d for d in range(-d_shift,0,1)]
            self.short_seq_shift = 12
            ii = range(20000-self.short_seq_shift)
            self.train_index = ii
            if mode=='test':
                self.train_index = range(len(self.date)-1348-self.short_seq_shift, len(self.date)-self.short_seq_shift-4)

        elif dataname=='BikeNYC':
            roadgraph = cv2.imread(image_path) 
            ret, thresh1 = cv2.threshold(roadgraph, 1, 1, cv2.THRESH_BINARY)
            hh,ww = 512,256
            thresh2 = cv2.resize(thresh1, (hh, ww), interpolation=cv2.INTER_NEAREST)
            self.roadgraph = thresh2[:,:,0]
            
            self.mean  = self.data[:3906].mean()
            self.var = self.data[:3906].std()
            self.data = (self.data-self.mean)/self.var
            d_shift =6
            self.long_seq_shift = [24*d for d in range(-d_shift,0,1)]
            self.short_seq_shift = 6
            ii = range(4150-self.short_seq_shift)
            self.train_index = ii
            if mode=='test':
                self.train_index = range(len(self.date)-244-self.short_seq_shift, len(self.date)-self.short_seq_shift-4)

        else data name=='TaxiNYC':
            roadgraph = cv2.imread(image_path) 
            ret, thresh1 = cv2.threshold(roadgraph, 1, 1, cv2.THRESH_BINARY)
            hh,ww = 640,384
            thresh2 = cv2.resize(thresh1, (ww, hh), interpolation=cv2.INTER_NEAREST)
            roadgraph = thresh2[:,:,0]
        
            self.mean  = self.data[:3928].mean()
            self.var = self.data[:3928].std()
            self.data = (self.data-self.mean)/self.var
            d_shift = 6
            self.long_seq_shift = [24*d for d in range(-d_shift,0,1)]
            self.short_seq_shift = 6
            ii = range(4172-self.short_seq_shift)
            self.train_index = ii
            if mode=='test':
                self.train_index = range(len(self.data)-244-self.short_seq_shift, len(self.data)-self.short_seq_shift-4)
        print('dataset loaded!', self.__len__(),'seqs')
        
    def __len__(self):
        return len(self.train_index)

    def __getitem__(self, index):
        index = self.train_index[index]
        short_seq_index = index+self.short_seq_shift
        short_seq = self.data[index:short_seq_index] 
        long_seq = [(self.data[i+short_seq_index]) for i in self.long_seq_shift]
        gt = self.data[short_seq_index:short_seq_index+1] 
        gt = gt.reshape((2, 32,32)) 
        
        sample = {'image':  np.concatenate((short_seq, long_seq),axis=0), 'label': gt,}
        
        sample= transforms.Compose([ToTensor()])(sample)

        return sample
    
def criterion(pred, labels):
    mseloss = torch.nn.MSELoss(reduction='sum')
    L1loss = torch.nn.L1Loss(reduction='mean')
    ps = pred.contiguous().view(-1).shape[0]
    loss = mseloss(pred, labels)/ps+0.1*L1loss(pred, labels)
    return loss

def cross_entropy2d(output, target_var , ignore_index=255, weight=None, size_average=True, batch_average=True , half = None):

    n, c, h, w = output.size()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    loss = criterion(output, target_var.long().squeeze(1))
    loss/= (h * w)
    loss /= n
    return loss

def cross_entropy1d(output, target_var , ignore_index=255, weight=None, size_average=True, batch_average=True , half = None):
    
    n, c = output.size()
    target_var = torch.topk(target_var, 1)[1].squeeze(1).view(n,1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='sum')
    loss = criterion(output, target_var.long().squeeze(1))
    loss/= c
    loss /= n
    return loss


class Measure(object):
    def __init__(self, preds, labels, var):
        super(Measure, self).__init__()
        self.var = var
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        arr = self.denorm([preds, labels])
        preds, labels = arr[0], arr[1]
        self.batch_size = labels.shape[0]
        self.rmse = self.masked_rmse_np(preds, labels, null_val=np.nan)
        self.mse = self.masked_mse_np(preds, labels, null_val=np.nan)
        self.mae = self.masked_mae_np(preds, labels, null_val=np.nan)
        self.mape = self.masked_mape_np(preds, labels, null_val=np.nan)

    def denorm(self, arr):
        dd=[]
        for item in arr:
            dd.append(item*self.var)
        return dd
    def _print(self):
        print('RMSE {rmse:.4f} MAE {mae:.4f}'.format(rmse=self.rmse, mae = self.mae))
    def masked_rmse_np(self, preds, labels, null_val=np.nan):
        return np.sqrt(self.masked_mse_np(preds=preds, labels=labels, null_val=null_val))

    def masked_mse_np(self, preds, labels, null_val=np.nan):
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.isnan(null_val):
                mask = ~np.isnan(labels)
            else:
                mask = np.not_equal(labels, null_val)
            mask = mask.astype('float32')
            mask /= np.mean(mask)
            rmse = np.square(np.subtract(preds, labels)).astype('float32')
            rmse = np.nan_to_num(rmse * mask)
            return np.mean(rmse)

    def masked_mae_np(self, preds, labels, null_val=np.nan):
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.isnan(null_val):
                mask = ~np.isnan(labels)
            else:
                mask = np.not_equal(labels, null_val)
            mask = mask.astype('float32')
            mask /= np.mean(mask)
            mae = np.abs(np.subtract(preds, labels)).astype('float32')
            mae = np.nan_to_num(mae * mask)
            return np.mean(mae)

    def masked_mape_np(self, preds, labels, null_val=np.nan):
        with np.errstate(divide='ignore', invalid='ignore'):
            if np.isnan(null_val):
                mask = ~np.isnan(labels)
            else:
                mask = np.not_equal(labels, null_val)
            mask = mask.astype('float32')
            mask /= np.mean(mask)
            mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
            mape = np.nan_to_num(mask * mape)
            return np.mean(mape)