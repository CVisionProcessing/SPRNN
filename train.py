import pickle as pkl
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
import glob
from sklearn.metrics import mean_squared_error,mean_absolute_error
import time
from tqdm import tqdm
from model import EGRNN
from utils import SeqDataset,criterion,Measure
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.backends.cudnn.enabled = True
parser = argparse.ArgumentParser()
parser.add_argument("data_name")
parser.add_argument("data_path")
parser.add_argument("image_path")
args = parser.parse_args()
if args.data_name=='TaxiBJ' or args.data_path=='BikeNYC' or args.image_path=='TaxiNYC':

    train(args.data_name, args.data_path, args.image_path)
else:
    raise Exception("数据集选择错误")

def train(dataname, data_path, image_path):
    batch_size = 64
    test_data = SeqDataset(dataname, data_path, image_path, mode='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size//2, shuffle=False, num_workers=8,  pin_memory=True)

    train_data = SeqDataset(dataname)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=8,  pin_memory=True, drop_last=True)
    steps = len(train_loader)
        
    rmse_1st = float("inf")
    model = SPRNN(batch_size)
    model.cuda()
    params = [
            {"params": model.parameters(), "lr": 0.001}
            ]
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.999), weight_decay=1e-5)
    epoches = 400
    RMSE =[]
    MAE =[]
    max_value=1

    for epoch in range(epoches):
        start_time = time.time()

        train_losses = 0
        train_rmse = 0
        roadgraph_var = torch.from_numpy(roadgraph).float()
        roadgraph_var = torch.autograd.Variable(roadgraph_var).cuda().view(1,1,512,512)
        roadgraph_var = roadgraph_var.repeat(batch_size,1,1,1)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=8,  pin_memory=True, drop_last=True)
        for st, sample in enumerate(train_loader):
            x_axis = (steps*epoch+st)*batch_size
            x_in_epoch = st*batch_size
            image, label = sample['image']/max_value, sample['label']/max_value
            wea, weather_label = sample['weather'], sample['weather_label']
            input_var = torch.autograd.Variable(image).cuda()
            target_var = torch.autograd.Variable(label).cuda()
            wea_var = torch.autograd.Variable(wea).cuda()
            wea_label_var = torch.autograd.Variable(weather_label).cuda()
            
            b,seq,f,_,_=input_var.shape

            output, predgraph = model(input_var, wea_var, input_var, roadgraph_var, mode='train')
            loss = criterion(output, target_var) 
            
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        btime = time.time()-start_time
        print('Epoch: {epoch} loss: {loss:.4f} time: {btime:.4f}'.format(epoch=epoch, \
              loss=train_losses/train_data.__len__()*var, btime=btime), end='\t')

        pred_last = None
        rmse = 0
        mse = 0
        mae=0
        
        roadgraph_var = torch.from_numpy(roadgraph).float()
        roadgraph_var = torch.autograd.Variable(roadgraph_var).cuda().view(1,1,512,512)
        roadgraph_var = roadgraph_var.repeat(batch_size//2,1,1,1)
        
        roadgraph_var = model(roadgraph_var, wea_var, roadgraph_var, roadgraph_var, mode='computegraph')
        for st, sample in enumerate(test_loader):
            x_axis = (steps*epoch+st)*batch_size
            x_in_epoch = st*batch_size
        
            image, label = sample['image']/max_value, sample['label']/max_value
            wea, weather_label = sample['weather'], sample['weather_label']
            input_var = torch.autograd.Variable(image).cuda()
            target_var = torch.autograd.Variable(label).cuda()
            wea_var = torch.autograd.Variable(wea).cuda()
            wea_label_var = torch.autograd.Variable(weather_label).cuda()
            
            b,seq,f,_,_=input_var.shape
            output = model(input_var, wea_var, input_var, roadgraph_var, mode='test')
            pred_last = output
            loss = criterion(output, target_var)
            mertric = Measure(output, target_var, train_data.var)
            
            mse+=mertric.mse*max_value*b
            mae+=mertric.mae*max_value*b
        RMSE.append(np.sqrt(mse/test_data.__len__()))
        MAE.append(mae/test_data.__len__())
        print('test RMSE {rmse:.4f} MAE {mae:.4f}'.format(rmse = RMSE[-1], mae = MAE[-1]))
        if RMSE[-1]<rmse_1st:
            model_name = 'best_epoch_{index}_RMSE_{rmse:.4f}_MAE_{mae:.4f}'.format(index = RMSE.index(np.min(RMSE)), \
                                                                                   rmse = RMSE[-1], mae = MAE[-1])
            torch.save(model, './models/'+model_name+'.pkl')
            pklarr = glob.glob('./models/*.pkl')
            for path in pklarr:
                if '{rmse:.4f}'.format(rmse = rmse_1st) in path:  
                    os.remove(path)  
            rmse_1st=RMSE[-1]
