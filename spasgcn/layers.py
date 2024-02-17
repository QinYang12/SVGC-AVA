import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

import sys, os, pdb, pickle
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(file_path)[0])
from spasgcn.basic_layers import *
from spasgcn.utils import load_graph_info, v_num_dict, timer

def Model_select(model='model0', *args, **kwargs):
    if model == 'model0':   
        return Model0(*args, **kwargs) #visual-only
    if model == 'modelR':   
        return ModelR(*args, **kwargs) #
    if model == 'model1':   
        return Model1(*args, **kwargs) #audio-only
    if model == 'model2':   
        return Model2(*args, **kwargs) #
    if model == 'model21':   
        return Model21(*args, **kwargs) #our model
    if model == 'model21H':   
        return Model21H(*args, **kwargs) #our model
    if model == 'model2c':   
        return Model2c(*args, **kwargs) #cossine
    if model == 'model2a':   
        return Model2a(*args, **kwargs) #without attention
    if model == 'model2w':   
        return Model2w(*args, **kwargs) #without multi-scale concanatenation
    if model == 'model2Sp':
        return Model2Sp(*args, **kwargs) #Using spectral cnn
    if model == 'model2HR':
        return Model2HR(*args, **kwargs) #Using spectral cnn
    if model == 'model22':   
        return Model22(*args, **kwargs) #
    if model == 'model4':   
        return Model4(*args, **kwargs) #
    if model == 'model5':   
        return Model5(*args, **kwargs) #

def Data_select(dataset='Chao', *args, **kwargs):
    if dataset == 'Chao':
        return Chao_Data(*args, **kwargs)
    if dataset == 'Qin':
        return Qin_Data(*args, **kwargs)
    return

class Model0(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model0, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        self.conv0 = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1 = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        self.conv2 = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, aem, Tlocation):
        assert x.dim() == 4  # B * T * 10242 * 2
        assert x.shape[2] == 10242
        conv0 = self.conv0(x)  # B * T * 10242 * 8
        spool0 = self.spool0(conv0) # B * T * 2562 * 8

        conv1 = self.conv1(spool0)  # B * T * 2562 * 16
        spool1 = self.spool1(conv1) # B * T * 642 * 16

        conv2 = self.conv2(spool1)  # B * T * 642 * 32

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, gl, in_channel, out_channel, kernel_s, stride=1, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.conv1 = SphereConv(gl, in_channel, out_channel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)   

    def forward(self, x):
        out = self.conv1(x)
        out = out + x
        out = F.relu(out)
        return out

class ModelR(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(ModelR, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        self.conv0 = SphereConv(gl, finchannel, 4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1 = ResidualBlock(gl-1, 4, 4, kernel_s, stride=1, *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        self.conv2 = ResidualBlock(gl-2, 4, 4, kernel_s, stride=1, *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, 4, 'max')
        self.conv7 = ResidualBlock(gl-1, 4+4, 8, kernel_s, stride=1, *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, 8, 'max')
        self.conv8 = ResidualBlock(gl, 8+4, 12, kernel_s, stride=1, *args, **kwargs)
        self.conv9 = SphereConv(gl, 12, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, aem, Tlocation):
        assert x.dim() == 4  # B * T * 10242 * 2
        assert x.shape[2] == 10242
        conv0 = self.conv0(x)  # B * T * 10242 * 8
        spool0 = self.spool0(conv0) # B * T * 2562 * 8

        conv1 = self.conv1(spool0)  # B * T * 2562 * 16
        spool1 = self.spool1(conv1) # B * T * 642 * 16

        conv2 = self.conv2(spool1)  # B * T * 642 * 32

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

class Model1(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model1, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        self.conv0 = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1 = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        self.conv2 = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert aem.dim() == 4  # B * T * 10242 * 2
        assert aem.shape[2] == 10242
        conv0 = self.conv0(aem)  # B * T * 10242 * 8
        spool0 = self.spool0(conv0) # B * T * 2562 * 8

        conv1 = self.conv1(spool0)  # B * T * 2562 * 16
        spool1 = self.spool1(conv1) # B * T * 642 * 16

        conv2 = self.conv2(spool1)  # B * T * 642 * 32

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

class Model2(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att = self.att1(conv1_f, conv1_a)


        conv2_f = self.conv2_f(spool1_f)  # B * T * 642 * 32

        unpool7 = self.unpool7(conv2_f) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

class Model21H(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model21H, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')

        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*4)
        self.spool2 = SpherePool(gl-2, 'max')

        self.conv3_f = SphereConv(gl-3, mchannel*4, mchannel*8, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv3_a = SphereConv(gl-3, mchannel*4, mchannel*8, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att3 = self_attention(mchannel*8, mchannel*8)
        self.spool3 = SpherePool(gl-3, 'max')
        
        self.conv4_f = SphereConv(gl-4, mchannel*8, mchannel*16, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv4_a = SphereConv(gl-4, mchannel*8, mchannel*16, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att4 = self_attention(mchannel*16, mchannel*8)

        self.unpool5 = SphereUnPool(gl-4, mchannel*16, 'max')
        self.conv5 = SphereConv(gl-3, mchannel*(16+8+8), mchannel*8, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool6 = SphereUnPool(gl-3, mchannel*8, 'max')
        self.conv6 = SphereConv(gl-2, mchannel*(8+4+4), mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att = self.att1(conv1_f, conv1_a)

        conv2_f = self.conv2_f(spool1_f)  # B * T * 2562 * 16
        spool2_f = self.spool2(conv2_f) # B * T * 642 * 16
        conv2_a = self.conv2_a(spool1_a)  # B * T * 2562 * 16
        spool2_a = self.spool2(conv2_a) # B * T * 642 * 16
        conv2_f_att, conv2_a_att = self.att2(conv2_f, conv2_a)

        conv3_f = self.conv3_f(spool2_f)  # B * T * 2562 * 16
        spool3_f = self.spool3(conv3_f) # B * T * 642 * 16
        conv3_a = self.conv3_a(spool2_a)  # B * T * 2562 * 16
        spool3_a = self.spool3(conv3_a) # B * T * 642 * 16
        conv3_f_att, conv3_a_att = self.att3(conv3_f, conv3_a)

        conv4_f = self.conv4_f(spool3_f)  # B * T/2 * 42 * 128
        conv4_a = self.conv4_a(spool3_a)  # B * T/2 * 42 * 128
        conv4_f_att, conv4_a_att = self.att4(conv4_f, conv4_a)
        conv4 = torch.cat((conv4_f_att, conv4_a_att), -1)

        unpool5 = self.unpool5(conv4) # B * T/4 * 2562 * 32
        cat5 = torch.cat((unpool5, conv3_f_att, conv3_a_att), -1) # B * T/4 * 2562 * 32+16
        conv5 = self.conv5(cat5)  # B * T/4 * 2562 * 16

        unpool6 = self.unpool6(conv5) # B * T/4 * 2562 * 32
        cat6 = torch.cat((unpool6, conv2_f_att, conv2_a_att), -1) # B * T/4 * 2562 * 32+16
        conv6 = self.conv6(cat6)  # B * T/4 * 2562 * 16

        unpool7 = self.unpool7(conv6) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att, conv1_a_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att, conv0_a_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

#to visual the visual-audio attention feature
class Model21(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model21, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242

        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att, v1v1, v1a2, v2a2, a2a2 = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att, _, _, _, _ = self.att1(conv1_f, conv1_a)


        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2_f_att, conv2_a_att, _, _, _, _ = self.att2(conv2_f, conv2_a)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att, conv1_a_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att, conv0_a_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        
        return output #, v1v1, v1a2, v2a2, a2a2
        #return conv0_f, conv0_a, conv0_f_att, conv0_a_att, conv1_f, conv1_a, conv1_f_att, conv1_a_att, output


"this is the final model for eccv"
class Model21_final(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model21_final, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att = self.att1(conv1_f, conv1_a)


        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2_f_att, conv2_a_att = self.att2(conv2_f, conv2_a)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att, conv1_a_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att, conv0_a_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        
        return output
        #return conv0_f, conv0_a, conv0_f_att, conv0_a_att, conv1_f, conv1_a, conv1_f_att, conv1_a_att, output

class Model2c(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2c, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*8, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(8+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        mask0 = self.simi(conv0_f, conv0_a) # B * T * 10242 (value:-1~1)
        conv0_f_att = torch.mul(conv0_f.permute(3, 0, 1, 2), mask0).permute(1,2,3,0)
        conv0_a_att = torch.mul(conv0_a.permute(3, 0, 1, 2), mask0).permute(1,2,3,0)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        mask1 = self.simi(conv1_f, conv1_a) # B * T * 2562 (value:-1~1)
        conv1_f_att = torch.mul(conv1_f.permute(3, 0, 1, 2), mask1).permute(1,2,3,0)
        conv1_a_att = torch.mul(conv1_a.permute(3, 0, 1, 2), mask1).permute(1,2,3,0)

        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        mask2 = self.simi(conv2_f, conv2_a) # B * T * 642 (value:-1~1)
        conv2_f_att = torch.mul(conv2_f.permute(3, 0, 1, 2), mask2).permute(1,2,3,0)
        conv2_a_att = torch.mul(conv2_a.permute(3, 0, 1, 2), mask2).permute(1,2,3,0)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att, conv1_a_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att, conv0_a_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

#without attention

class Model2a(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2a, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16

        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2 = torch.cat((conv2_f, conv2_a), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f, conv1_a), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f, conv0_a), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

#without multi-scale visual-audio attention

class Model2w(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2w, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16

        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2_f_att, conv2_a_att = self.att2(conv2_f, conv2_a)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        conv7 = self.conv7(unpool7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        conv8 = self.conv8(unpool8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

class Model2Sp(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2Sp, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SpectralConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SpectralConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SpectralConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SpectralConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SpectralConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SpectralConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SpectralConv(gl-1, mchannel*(4+2+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SpectralConv(gl, mchannel*(2+1+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SpectralConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att = self.att1(conv1_f, conv1_a)


        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2_f_att, conv2_a_att = self.att2(conv2_f, conv2_a)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att, conv1_a_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att, conv0_a_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output

class Model2HR(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model2HR, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0 = SpectralConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1 = SpectralConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool1 = SpherePool(gl-1, 'max')
        self.conv2 = SpectralConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool2 = SpherePool(gl-2, 'max')
        self.conv3 = SpectralConv(gl-3, mchannel*4, mchannel*8, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.spool3 = SpherePool(gl-3, 'max')
        self.conv4 = SpectralConv(gl-4, mchannel*8, mchannel*16, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)     

        self.unpool5 = SphereUnPool(gl-4, mchannel*16, 'max')
        self.conv5 = SpectralConv(gl-3, mchannel*(16+8), mchannel*8, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool6 = SphereUnPool(gl-3, mchannel*8, 'max')
        self.conv6 = SpectralConv(gl-2, mchannel*(8+4), mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SpectralConv(gl-1, mchannel*(4+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SpectralConv(gl, mchannel*(2+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        
        self.conv9 = SpectralConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0 = self.conv0(frame)  # B * T * 10242 * 8
        spool0 = self.spool0(conv0) # B * T * 2562 * 8        
        conv1 = self.conv1(spool0)  # B * T * 2562 * 16
        spool1 = self.spool1(conv1) # B * T * 642 * 16
        conv2 = self.conv2(spool1)  # B * T * 10242 * 8
        spool2 = self.spool2(conv2) # B * T * 2562 * 8        
        conv3 = self.conv3(spool2)  # B * T * 2562 * 16
        spool3 = self.spool3(conv3) # B * T * 642 * 16
        conv4 = self.conv4(spool3)  # B * T/2 * 42 * 128

        unpool5 = self.unpool5(conv4) # B * T/4 * 2562 * 32
        cat5 = torch.cat((unpool5, conv3), -1) # B * T/4 * 2562 * 32+16
        conv5 = self.conv5(cat5)  # B * T/4 * 2562 * 16
        unpool6 = self.unpool6(conv5) # B * T/4 * 2562 * 32
        cat6 = torch.cat((unpool6, conv2), -1) # B * T/4 * 2562 * 32+16
        conv6 = self.conv6(cat6)  # B * T/4 * 2562 * 16
        unpool7 = self.unpool7(conv6) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16
        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output


class Model22(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model22, self).__init__()
        gl, finchannel, ainchannel, mchannel = graph_level, kwargs['finchannel'], kwargs['ainchannel'], kwargs['mchannel']
        
        self.conv0_f = SphereConv(gl, finchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv0_a = SphereConv(gl, ainchannel, mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att0 = self_attention(mchannel, mchannel)
        self.spool0 = SpherePool(gl, 'max')
        self.conv1_f = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv1_a = SphereConv(gl-1, mchannel, mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att1 = self_attention(mchannel*2, mchannel*2)
        self.spool1 = SpherePool(gl-1, 'max')
        
        self.conv2_f = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv2_a = SphereConv(gl-2, mchannel*2, mchannel*4, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.att2 = self_attention(mchannel*4, mchannel*2)

        self.unpool7 = SphereUnPool(gl-2, mchannel*4, 'max')
        self.conv7 = SphereConv(gl-1, mchannel*(4+2), mchannel*2, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.unpool8 = SphereUnPool(gl-1, mchannel*2, 'max')
        self.conv8 = SphereConv(gl, mchannel*(2+1), mchannel, kernel_s, post_layers=nn.Sequential(nn.ReLU(inplace=True)), *args, **kwargs)
        self.conv9 = SphereConv(gl, mchannel, 1, kernel_s, *args, **kwargs)
        
        self.tpool = TemporalPool(pool_t)
        self.softmax = nn.Softmax(dim=-1)
        self.simi = nn.CosineSimilarity(dim=-1)

    def forward(self, frame, aem, Tlocation):
        aem = torch.unsqueeze(aem, -1)
        assert frame.dim() == 4  # B * T * 10242 * 2
        assert aem.dim() == 4
        assert frame.shape[2] == 10242
        conv0_f = self.conv0_f(frame)  # B * T * 10242 * 8
        spool0_f = self.spool0(conv0_f) # B * T * 2562 * 8        
        conv0_a = self.conv0_a(aem)  # B * T * 10242 * 8
        spool0_a = self.spool0(conv0_a) # B * T * 2562 * 8
        conv0_f_att, conv0_a_att = self.att0(conv0_f, conv0_a)

        conv1_f = self.conv1_f(spool0_f)  # B * T * 2562 * 16
        spool1_f = self.spool1(conv1_f) # B * T * 642 * 16
        conv1_a = self.conv1_a(spool0_a)  # B * T * 2562 * 16
        spool1_a = self.spool1(conv1_a) # B * T * 642 * 16
        conv1_f_att, conv1_a_att = self.att1(conv1_f, conv1_a)


        conv2_f = self.conv2_f(spool1_f)  # B * T/2 * 42 * 128
        conv2_a = self.conv2_a(spool1_a)  # B * T/2 * 42 * 128
        conv2_f_att, conv2_a_att = self.att2(conv2_f, conv2_a)
        conv2 = torch.cat((conv2_f_att, conv2_a_att), -1)

        unpool7 = self.unpool7(conv2) # B * T/4 * 2562 * 32
        cat7 = torch.cat((unpool7, conv1_f_att), -1) # B * T/4 * 2562 * 32+16
        conv7 = self.conv7(cat7)  # B * T/4 * 2562 * 16

        unpool8 = self.unpool8(conv7) # B * T/4 * 10242 * 16
        cat8 = torch.cat((unpool8, conv0_f_att), -1) # B * T/4 * 10242 * 16+8
        conv8 = self.conv8(cat8) # B * T/4 * 10242 * 8

        conv9 = self.conv9(conv8) # B * T/4 * 10242 * 1
        conv9 = torch.squeeze (conv9, 1) # B * 10242 * 1
        conv9 = torch.squeeze (conv9, -1) # B * 10242
        output = self.softmax(conv9)
        return output



class Model4(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model4, self).__init__()
        self.block1 = Model0(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)
        self.block2 = Model1(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)
        self.block3 = Model21(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)

    def forward(self, x, aem, Tlocation):
        assert x.dim() == 4  # B * T * 10242 * 2
        assert x.shape[2] == 10242
        s1 = self.block1(x, aem, Tlocation)  # B * 10242
        s2 = self.block2(x, aem, Tlocation)
        s3 = self.block3(x, aem, Tlocation)
        s = 0.2*s1+0.3*s2+0.5*s3 # B * 10242
        return s

class Model5(nn.Module):
    @timer
    def __init__(self, graph_level=5, kernel_s=7, kernel_t=3, pool_t=2, *args, **kwargs):
        super(Model5, self).__init__()
        self.block1 = Model0(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)
        self.block2 = Model1(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)
        self.block3 = Model21(graph_level=graph_level, kernel_s=kernel_s, kernel_t=kernel_t, pool_t=pool_t, *args, **kwargs)
        self.finalrate = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax()
        
    def forward(self, x, aem, Tlocation):
        assert x.dim() == 4  # B * T * 10242 * 2
        assert x.shape[2] == 10242
        s1 = self.block1(x, aem, Tlocation)  # B * 10242
        s2 = self.block2(x, aem, Tlocation)
        s3 = self.block3(x, aem, Tlocation)
        rate = self.softmax(self.finalrate)
        s = rate[0]*s1+rate[1]*s2+rate[2]*s3 # B * 10242
        #print(rate)
        return s


class Chao_Data(Dataset):
    def __init__(self, root='data', len_snippet=5, act='train', graph_level=5, gaussian='small'):
        super().__init__()
        self.act = act
        self.imgs, self.labels, self.aems = [], [], []
        imgs, labels, aems = self.data_prepare(root, len_snippet, act, graph_level)
        
        # img_num *frames* v_num * inc, img_num*v_num
        with torch.no_grad():
            for i in range(len(labels)):
                self.imgs.append(torch.tensor(imgs[i]).float())  # img_num *frames* v_num * inc, img_num*v_num
                self.labels.append(labels[i])
                self.aems.append(torch.tensor(aems[i]).float())

    def data_prepare(self, root, len_snippet, act, graph_level):
        datapath = os.path.join(root, 'Chao', f'Chaomel_{act}_G{graph_level}_L{len_snippet}')
        try:
            f_read = open(datapath+'.pkl', 'rb')
            data = pickle.load(f_read)
            f_read.close()
            imgs = data['img']  # img_num * frames * v_num * inc
            labels = data['label']  # img_num
            aems = data['aem']
            # imgs, labels, aems = imgs[:100], labels[:100], aems[:100]  # 测试用，只取一小点
            assert len(imgs) == len(labels)
            return imgs, labels, aems
        except (FileNotFoundError, AssertionError):
            print('no pkl')
            pdb.set_trace()


    def __getitem__(self, index):
        return self.imgs[index], self.labels[index], self.aems[index], torch.tensor(0), torch.tensor(0)

    def __len__(self):
        return len(self.labels)


class Qin_Data(Dataset):
    def __init__(self, root='data', len_snippet=5, act='train', graph_level=5, gaussian='large'):
        super().__init__()
        self.act = act
        self.imgs, self.labels, self.aems, self.index_l, self.index_f = [], [], [], [], []
        imgs, labels, aems, index_l, index_f = self.data_prepare(root, len_snippet, act, graph_level, gaussian)
        
        # img_num *frames* v_num * inc, img_num*v_num
        with torch.no_grad():
            for i in range(len(labels)):
                self.imgs.append(torch.tensor(imgs[i]).float())  # img_num *frames* v_num * inc, img_num*v_num
                self.labels.append(labels[i])
                self.aems.append(torch.tensor(aems[i]).float())
                self.index_l.append(index_l[i])
                self.index_f.append(index_f[i])

    def data_prepare(self, root, len_snippet, act, graph_level, gaussian):
        datapath = os.path.join(root, 'Qin', f'Qinmel_{act}_G{graph_level}_L{len_snippet}_R5_{gaussian}_random')
        try:
            f_read = open(datapath+'.pkl', 'rb')
            data = pickle.load(f_read)
            f_read.close()
            imgs = data['img']  # img_num * frames * v_num * inc
            labels = data['label']  # img_num
            aems = data['aem']
            index_l = data['index_l']
            index_f = data['index_f']
            # imgs, labels, aems = imgs[:100], labels[:100], aems[:100]  # 测试用，只取一小点
            assert len(imgs) == len(labels)
            return imgs, labels, aems, index_l, index_f
        except (FileNotFoundError, AssertionError):
            print('no pkl')
            pdb.set_trace()


    def __getitem__(self, index):
        return self.imgs[index], self.labels[index], self.aems[index], self.index_l[index], self.index_f[index]

    def __len__(self):
        return len(self.labels)

class Qin_Data_noidx(Dataset):
    def __init__(self, root='data', len_snippet=5, act='train', graph_level=5):
        super().__init__()
        self.act = act
        self.imgs, self.labels, self.aems = [], [], []
        imgs, labels, aems = self.data_prepare(root, len_snippet, act, graph_level)
        
        # img_num *frames* v_num * inc, img_num*v_num
        with torch.no_grad():
            for i in range(len(labels)):
                self.imgs.append(torch.tensor(imgs[i]).float())  # img_num *frames* v_num * inc, img_num*v_num
                self.labels.append(labels[i])
                self.aems.append(torch.tensor(aems[i]).float())


    def data_prepare(self, root, len_snippet, act, graph_level):
        datapath = os.path.join(root, 'Qin', f'Qin_{act}_G{graph_level}_L{len_snippet}_R5')
        try:
            f_read = open(datapath+'.pkl', 'rb')
            data = pickle.load(f_read)
            f_read.close()
            imgs = data['img']  # img_num * frames * v_num * inc
            labels = data['label']  # img_num
            aems = data['aem']
            # imgs, labels, aems = imgs[:100], labels[:100], aems[:100]  # 测试用，只取一小点
            assert len(imgs) == len(labels)
            return imgs, labels, aems
        except (FileNotFoundError, AssertionError):
            print('no pkl')
            pdb.set_trace()


    def __getitem__(self, index):
        return self.imgs[index], self.labels[index], self.aems[index]

    def __len__(self):
        return len(self.labels)
