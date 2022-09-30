import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class GRUCell(nn.Module):
    def __init__(self, batch, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_out_size = hidden_size
        self.bias = bias
        self.conv1dr = nn.Conv2d(self.hidden_size+32, self.hidden_size*2, kernel_size=3, padding=1)
        self.conv1dc = nn.Conv2d(self.hidden_size+32, self.hidden_size, kernel_size=3, padding=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden_state, nodes_num):        
        x_ = F.sigmoid(self.rgc(x, hidden_state, nodes_num))
        r, u = x_.chunk(2, 1)
        r_state = r * hidden_state
        
        c = F.tanh(self.cgc(x, r_state, nodes_num))

        new_h = u * hidden_state + (1 - u) * c
        
        return new_h
    
    def cgc(self, x, hidden_state, nodes_num):
        b,f,n,_ = x.shape
        x = x.contiguous().view(b,  -1, n,n)
        hidden_state = hidden_state.view(b, -1, n,n)
        x_s = torch.cat([x, hidden_state], 1)
        x= self.conv1dc(x_s)
        
        return x
    def rgc(self, x, hidden_state, nodes_num):
        
        b,f,n,_ = x.shape
        x = x.contiguous().view(b,  -1, n,n)
        hidden_state = hidden_state.view(b, -1, n,n)
        x_s = torch.cat([x, hidden_state], 1)
        x= self.conv1dr(x_s)
        
        return x
        
class GRUModel(nn.Module):
    def __init__(self, nodes, batch, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.nodes = nodes
        self.layer_dim = layer_dim
        self.convmerge = nn.Sequential(nn.Conv2d(2+64, 32, kernel_size=3, padding=1), nn.ReLU())
        self.gru_cell = GRUCell(batch, input_dim, hidden_dim)
        self.fuout = nn.Sequential(nn.Conv2d(hidden_dim, 2, kernel_size=3, stride=1, padding=1,bias=False))
                                   
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, wea_var, roadgraph_var):
        b,seq,f,n,_ = x.shape

        nodes_num = n*n
        hn = torch.autograd.Variable(torch.zeros(b, self.hidden_dim, n,n).cuda())
        wea_hn = torch.autograd.Variable(torch.zeros(b, 32).cuda())
        wea_hn_c = torch.autograd.Variable(torch.zeros(b, 32).cuda())
        hnout = torch.autograd.Variable(torch.zeros(b, self.hidden_dim, n,n).cuda())
                                   
        outs = []
        for seq in range(x.size(1)):
            tmp = self.convmerge(torch.cat([x[:,seq], roadgraph_var], 1))
            hn = self.gru_cell(tmp, hn, nodes_num)
            outs.append(hn)
        
        out = outs[-1]

        return out

class Graphfeat(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Graphfeat, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.sigmoid_atten = nn.Sigmoid()
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                 )
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                 )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True)
                                 )
        self.conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1,bias=False),
                                   nn.BatchNorm2d(128),nn.ReLU(inplace=True)
                                 )
        self.conv55 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
                                   nn.BatchNorm2d(128),
                                 )
        self.conv54 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                 )
        self.conv43 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,bias=False),
                                   nn.BatchNorm2d(64),nn.ReLU(inplace=True)
                                 )

    def forward(self, x, mode='train'):
        raw_size = x.size()[2:]
        x0=self.conv1(x)
        x=self.maxpool(x0)
        
        x1=self.conv2(x)
        x2=self.conv3(x1)    #32x32
        x3=self.conv4(x2)
        x4=self.conv5(x3)
        atten = F.avg_pool2d(x4, (6, 6))
        atten = F.interpolate(atten, x4.size()[2:], mode='nearest')
        atten=self.conv55(x4)
        atten = self.sigmoid_atten(atten)
        x4 = x4*atten
        x = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x=self.conv54(torch.cat((x,x3),1))
        x = F.interpolate(x, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x2=self.conv43(torch.cat((x,x2),1))
        
        if mode!='train':
            return x2,_
        return x2,_
    
class SPRNN(nn.Module):
    def __init__(self, batch, nodes=32*32, input_dim=32*32*2, hidden_dim=64, layer_dim=1, output_dim=2):
        print("Constructing SPRNN model...")
        super(SPRNN, self).__init__()
        
        self.grumodel = GRUModel(nodes, batch, input_dim, hidden_dim, layer_dim, output_dim).cuda()
        self.graphfeat = Graphfeat(in_chan = 1, out_chan =  64)
        self.conv = nn.Sequential(nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1,),
                                 )
        self.wea_conv = nn.Sequential(nn.Linear(32, 32),nn.ReLU(inplace=True))
        self.wea_out1 = nn.Sequential(nn.Linear(32, 17))
        self.wea_out2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(1)
    def _makegrumodels(self, grunums, input_dim, hidden_dim, layer_dim, output_dim):
        for i in range(grunums):
            self.grumodels.append(GRUModel(input_dim, hidden_dim, layer_dim, output_dim).cuda())
            
    def forward(self, input_, wea_var, A, roadgraph_var, mode = 'train'):
        if mode=='train':
            roadgraph_var, predgraph = self.graphfeat(roadgraph_var)
        elif mode=='computegraph':
            roadgraph_var, predgraph = self.graphfeat(roadgraph_var, mode)
            return roadgraph_var
 
        b,seq,f,n,_ = input_.shape
        tem = self.grumodel(input_, wea_var, roadgraph_var)  # in (b,seq,f,nodes)
        
        out = self.conv(tem)
        out = out.view(b,f,n,n)

        if mode=='train':
            return out, predgraph     
        return out
