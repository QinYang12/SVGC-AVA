import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
import scipy.io
import pdb, sys, os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(file_path)[0])
from spasgcn.utils import gen_indexes, show_memory, timer, v_num_dict


class Interploate(Function):
    @staticmethod
    def forward(ctx, input, itp_mat):
        ctx.save_for_backward(itp_mat)  # v_num * 7 * ks
        return torch.matmul(input, itp_mat)

    @staticmethod
    def backward(ctx, grad_output):
        itp_mat,  = ctx.saved_tensors
        itp_mat = itp_mat.permute(0, 2, 1)
        return torch.matmul(grad_output, itp_mat), None

class self_attention(nn.Module):
    def __init__(self, Cin, Cout ,*args, **kwargs):
        super(self_attention, self).__init__()
        self.wv1 = nn.Linear(Cin, Cout, bias=False)
        self.wk1 = nn.Linear(Cin, Cout, bias=False)
        self.wq1 = nn.Linear(Cin, Cout, bias=False)
        self.wv2 = nn.Linear(Cin, Cout, bias=False)
        self.wk2 = nn.Linear(Cin, Cout, bias=False)
        self.wq2 = nn.Linear(Cin, Cout, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tensor1, tensor2):
        v1 = self.wv1(tensor1) # B, T, N, Cout 
        k1 = self.wk1(tensor1) 
        q1 = self.wq1(tensor1)

        v2 = self.wv2(tensor2)
        k2 = self.wk2(tensor2)
        q2 = self.wq2(tensor2)
        
        a11 = torch.squeeze(torch.matmul(torch.unsqueeze(k1, -2), torch.unsqueeze(q1, -1)), -1)
        a12 = torch.squeeze(torch.matmul(torch.unsqueeze(k2, -2), torch.unsqueeze(q1, -1)), -1)
        a1 = torch.cat((a11, a12), -1)
        a1 = self.softmax(a1) #B, T, N, 2

        a21 = torch.squeeze(torch.matmul(torch.unsqueeze(k1, -2), torch.unsqueeze(q2, -1)), -1)
        a22 = torch.squeeze(torch.matmul(torch.unsqueeze(k2, -2), torch.unsqueeze(q2, -1)), -1)
        a2 = torch.cat((a21, a22), -1)
        a2 = self.softmax(a2) #B, T, N, 2
        
        return v1*torch.unsqueeze(a1[:, :, :, 0], -1)+v2*torch.unsqueeze(a1[:, :, :, 1], -1), v1*torch.unsqueeze(a2[:, :, :, 0], -1)+v2*torch.unsqueeze(a2[:, :, :, 1], -1), v1*torch.unsqueeze(a1[:, :, :, 0], -1), v2*torch.unsqueeze(a1[:, :, :, 1], -1), v1*torch.unsqueeze(a2[:, :, :, 0], -1), v2*torch.unsqueeze(a2[:, :, :, 1], -1)  

class self_attention_mhead(nn.Module):
    def __init__(self, inchannel ,*args, **kwargs):
        super(self_attention_mhead, self).__init__()
        self.linear = nn.linear()
    def forward(self, tensorv, tensorq):
        v = self.linear()
        k = self.linear()
        q = self.linear()
        v_tensorv = v*tensorv
        k_tensorv = k*tensorv
        q_tensorq = q*tensorq
        return v_tensorv*(k_tensorv*q_tensorq)

class TemporalPool(nn.Module):
    def __init__(self, kernel_size, *agrs, **kwargs):
        super(TemporalPool, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size) #valiad
    def forward(self, tensor):
        assert tensor.dim() == 4
        B, T, N, C = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1) # B, N, C, T
        tensor = torch.reshape(tensor, (B*N, C, T))
        tensor = self.pool(tensor)
        tensor = torch.reshape(tensor, (B, N, C, tensor.shape[-1]))
        tensor = tensor.permute(0, 3, 1, 2) # B, T, N, C
        return tensor

class TemporalConv(nn.Module):
    def __init__(self, input_channel, 
                 output_channel, 
                 kernel_size, 
                 *agrs, **kwargs):
        super(TemporalConv, self).__init__()
        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, padding=1, padding_mode="circular") #to check
    def forward(self, tensor):
        assert tensor.dim() == 4
        B, T, N, C = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1) # B, N, C, T
        tensor = torch.reshape(tensor, (B*N, C, T))
        tensor = self.conv(tensor)
        tensor = torch.reshape(tensor, (B, N, C, tensor.shape[-1]))
        tensor = tensor.permute(0, 3, 1, 2) # B, T, N, C
        return tensor

class SphereConv(nn.Module):
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=9,
                 pre_layers=None,
                 post_layers=None,
                 *args, **kwargs):
        super(SphereConv, self).__init__()
        self.index, itp_mat = gen_indexes(graph_level, 'conv', kernel_size, *args, **kwargs)
        self.index = self.index.cuda()
        self.register_buffer("itp_mat", itp_mat)  # v_num * 7 * kernel_size
        self.conv = nn.Conv2d(input_channel, output_channel, (1, kernel_size))
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.pre_layers = pre_layers
        self.post_layers = post_layers

    def interpolate_dumb(self, tensor):
        # tensor  # batch_size * input_channel * v_num
        if self.kernel_size == 1:
            return tensor.unsqueeze(-1)  # batch_size * input_channel * v_num * ks(1)

        # tensor  # batch_size * input_channel * v_num
        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num * 7

        tensor = tensor.unsqueeze(-2)  # bs * inc * v_num * 1 * 7
        # self.itp_mat  # v_num * 7 * ks
        # out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks
        out_tensor = Interploate.apply(tensor, self.itp_mat)
        tensor = out_tensor.squeeze(-2)  # bs * inc * v_num * ks
        return tensor

    def interpolate_smart(self, tensor):
        # tensor  # batch_size * input_channel * v_num
        if self.kernel_size == 1:
            return tensor.unsqueeze(-1)  # batch_size * input_channel * v_num * ks(1)
        tensor1 = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor2 = tensor1[self.index]  # v_num * 7 * bs * inc
        tensor = tensor2.permute(2, 3, 0, 1)  # bs * inc * v_num * 7

        tensor = tensor.unsqueeze(-2)  # bs * inc * v_num * 1 * 7
        # self.itp_mat  # v_num * 7 * ks
        # out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks
        # tensor = out_tensor.squeeze(-2)  # bs * inc * v_num * ks
        # return tensor
        # 理论上是如上式子，但是这样会申请一个 bs * inc * v_num * 7 * ks 的空间，瞬时内存开销巨大
        # 而实际上只需要 1/7 的空间，所以可以把这个运算拆分成几个step计算，用时间换空间
        # 思路：直接把 inc 拆分计算

        out_shape = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1, self.kernel_size]
        out_tensor = torch.zeros(out_shape, device=tensor.device)  # bs * inc * v_num * 1 * ks
        mem_cost = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[4] * 4 / 1024 / 1024
        mem_goal = 1000
        inc_break = int(mem_cost//mem_goal) + 1
        inc_pb = tensor.shape[0]//inc_break + 1  # input_channel per break
        for cpb in range(inc_break):
            d = cpb * inc_pb
            u = (cpb+1) * inc_pb
            out_tensor[:, d:u] = Interploate.apply(tensor[:, d:u], self.itp_mat)

        out_tensor = out_tensor.squeeze(-2)  # bs * inc * v_num * ks
        return out_tensor

    def interpolate(self, tensor):
        # tensor  # batch_size * input_channel * v_num
        if self.kernel_size == 1:
            return tensor.unsqueeze(-1)  # batch_size * input_channel * v_num * ks(1)

        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num * 7

        tensor = tensor.unsqueeze(-2)  # bs * inc * v_num * 1 * 7
        # self.itp_mat  # v_num * 7 * ks
        # out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks
        # tensor = out_tensor.squeeze(-2)  # bs * inc * v_num * ks
        # return tensor
        # 理论上是如上式子，但是这样会申请一个 bs * inc * v_num * 7 * ks 的空间，瞬时内存开销巨大
        # 而实际上只需要 1/7 的空间，所以可以把这个运算拆分成几个step计算，用时间换空间
        # 思路：把 bs 和 inc 拆分计算

        out_shape = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1, self.itp_mat.shape[-1]]
        mem_waste = (tensor.shape[0]-1)*tensor.shape[1]*tensor.shape[2]*7*self.itp_mat.shape[-1]*4/1024**2
        mem_batch = tensor.shape[1]*tensor.shape[2]*7*self.itp_mat.shape[-1]*4/1024**2
        mem_out = tensor.shape[0]*tensor.shape[1]*tensor.shape[2]*self.itp_mat.shape[-1]
        # mem_goal = 350  # yq1
        mem_goal = 1000

        # batch: 第一个 batch 计算的时候瞬时内存开销, waste: 其余 batch 计算的时候的内存开销
        # batch > 1000, waste > 1000: 多 batch, 每个 batch 都大, batch 和 inc 分开计算
        # batch > 1000, waste < 1000: 单 batch, 每个 batch 都大, batch 和 inc 分开计算
        # batch < 1000, batch + waste > 1000: 多个 batch, 每个 batch < 1000, batch 分开计算
        # batch < 1000, batch + waste < 1000: 直接计算
        # print(f"Max memory cost: {mem_batch:.1f}MB per batch, {(mem_batch+mem_waste):.1f}MB in total; ", end='')
        # 若不拆分：
        # mem_waste, mem_batch = 0, 0

        if mem_batch > mem_goal:  # 前两种
            out_tensor = torch.zeros(out_shape, device=tensor.device)  # bs * inc * v_num * 1 * ks
            inc_break = int(mem_batch//mem_goal) + 1
            inc_pb = tensor.shape[1]//inc_break + 1  # input_channel per break
            # print(f"Actual memory cost: {inc_pb*tensor.shape[2]*7*self.itp_mat.shape[-1]*4/1024**2:.1f}MB")
            for bs in range(tensor.shape[0]):
                for cpb in range(inc_break):
                    d = cpb * inc_pb
                    u = (cpb+1) * inc_pb
                    out_tensor[bs][d:u] = Interploate.apply(tensor[bs][d:u], self.itp_mat)
                    # out_tensor[bs][d:u] = torch.matmul(tensor[bs][d:u], self.itp_mat)  # 1 * inc_pb * v_num * 1 * ks
        elif mem_batch + mem_waste > mem_goal:  # 第三种
            out_tensor = torch.zeros(out_shape, device=tensor.device)  # bs * inc * v_num * 1 * ks
            bs_pb = int(mem_goal/mem_batch)
            # print(f"Actual memory cost: {mem_batch*bs_pb}MB")
            bs_break = tensor.shape[0]//bs_pb + 1
            for bs in range(bs_break):
                d = bs * bs_pb
                u = (bs+1) * bs_pb
                out_tensor[d:u] = Interploate.apply(tensor[d:u], self.itp_mat)
                # out_tensor[d:u] = torch.matmul(tensor[d:u], self.itp_mat)  # 1 * inc * v_num * 1 * ks
        else:  # 直接算, 小模型，省去for循环
            # print(f"you are running at max cost")
            out_tensor = Interploate.apply(tensor, self.itp_mat)
            # out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks

        out_tensor = out_tensor.squeeze(-2)  # bs * inc * v_num * ks
        return out_tensor

    def forward(self, tensor):
        if self.pre_layers:
            tensor = self.pre_layers(tensor)
        assert tensor.dim() == 4  # batch_size * T *  v_num * inc
        B, T, N, C = tensor.shape
        tensor1 = torch.reshape(tensor, (B*T, N, C))
        tensor2 = tensor1.permute(0, 2, 1) # bs * inc * v_num
        tensor3 = self.interpolate_smart(tensor2)  # bs * inc * v_num * kernel_size
        tensor4 = self.conv(tensor3)  # batch_size * output_channel * v_num * 1
        tensor5 = tensor4.squeeze(-1)  # batch_size * output_channel * v_num
        if self.post_layers:
            tensor6 = self.post_layers(tensor5)
            tensor7 = torch.reshape(tensor6, (B, T, self.output_channel, N ))
            output = tensor7.permute(0, 1, 3, 2) # B T N C
        else:
            tensor6 = torch.reshape(tensor5, (B, T, self.output_channel, N ))
            output = tensor6.permute(0, 1, 3, 2) # B T N C

        if False:#True:# output[0,0].max(0)[0].min()==0:
            print("0 channel")
            print("tensor  max", tensor[0,0].max(0)[0])
            print("tensor1 max", tensor[0,0].max(0)[0])
            print("tensor3 max", tensor3[0,:,:,0].max(1)[0])
            print("tensor4 max", tensor4[0].max(1)[0])
            if self.post_layers:
                print(self.post_layers)
                print("tensor8 max", tensor8[0,0].max(0)[0])
            print("output  max", output[0,0].max(0)[0])
            # pdb.set_trace()
        
        return output

class SpectralConv(nn.Module):
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=20,
                 pre_layers=None,
                 post_layers=None,
                 *args, **kwargs):
        super(SpectralConv, self).__init__()
        self.Ks = kernel_size     
        self.Cout = output_channel   
        self.L = scipy.io.loadmat("/home/yq/Audio/Audio_visual/spasgcn/graph_laplacian.mat")['graph_laplacian'][0][5-graph_level]
        self.L = scipy.sparse.csr_matrix(self.L)
        self.L = self.rescale_L(self.L)
        self.L = self.L.tocoo()
        indices = torch.LongTensor(np.row_stack((self.L.row, self.L.col)))
        data = torch.FloatTensor(self.L.data)
        size = torch.Size(self.L.shape)
        self.L = torch.sparse.FloatTensor(indices, data, size).cuda()
        self.W = nn.Linear(input_channel*kernel_size, output_channel, bias=False)
    
    def rescale_L(self, L, lmax=2):
        """Rescale the Laplacian eigenvalues in [-1,1]."""
        M, M = L.shape
        I = scipy.sparse.identity(M, format='csr', dtype=L.dtype)
        L /= lmax / 2
        L -= I
        return L

    def concat(self, x, x_):
        x_ = x_.unsqueeze(0) # 1 * N * BTC
        return torch.cat((x, x_), 0) # K * N * BTC

    def forward(self, tensor):
        assert tensor.dim() == 4
        B, T, N, C = tensor.shape
        tensor = torch.reshape(tensor, (B*T, N, C))
        tensor = tensor.permute(1, 2, 0) # N C B*T
        x0 = torch.reshape(tensor, (N, B*T*C)) 
        x = x0.unsqueeze(0) # 1 N BTC

        if self.Ks > 1:
            x1 = torch.matmul(self.L, x0)
            x = self.concat(x, x1)
        for k in range(2, self.Ks):
            x2 = 2*torch.matmul(self.L, x1)-x0
            x = self.concat(x, x2)
            x0, x1 = x1, x2
        x = torch.reshape(x, [self.Ks, N, C, B*T])
        x = x.permute(3, 1, 2, 0) # B*T N C Ks
        x = torch.reshape(x, [B*T*N, C*self.Ks])
        x = self.W(x) #B*T*N Cout
        return torch.reshape(x, (B, T, N, self.Cout))

class ChebyConv(nn.Module):
    # utilize the dgl.chebConv
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=9,
                 pre_layers=None,
                 post_layers=None,
                 *args, **kwargs):
        super(ChebyConv, self).__init__()

    def forward(self, tensor):
        assert tensor.dim() == 4
        B, T, N, C = tensor.shape
        tensor = tensor.reshape(tensor, (B*T, N, C))
        tensor = tensor.permute(1, 2, 0)
        tensor = tensor.reshape(tensor, (N, B*T*C))

class SpherePool(nn.Module):
    def __init__(self, graph_level,
                 method='sample',
                 *agrs, **kwargs):
        super(SpherePool, self).__init__()
        self.index = gen_indexes(graph_level, 'pool')
        self.method = method

    def forward(self, tensor):
        assert tensor.dim() == 4  # batch_size * input_channel * v_num
        B, T, N, C = tensor.shape
        tensor = torch.reshape(tensor, (B*T, N, C))  # B V I
        tensor = tensor.permute(1, 0, 2)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num' * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num' * 7

        if self.method == 'sample':
            tensor = tensor[:, :, :, -1]
        elif self.method == 'max':
            tensor = tensor.max(dim=-1).values
        elif self.method == 'mean':
            tensor = tensor.mean(dim=-1).values

        tensor = torch.reshape(tensor, (B, T, C, -1 ))
        tensor = tensor.permute(0, 1, 3, 2) # B T N C

        return tensor  # bs * inc * v_num'


class SphereUnPool(nn.Module):
    def __init__(self, graph_level,input_channel,
                 method='sample',
                 *agrs, **kwargs):
        super(SphereUnPool, self).__init__()
        self.input_size, self.output_size = gen_indexes(graph_level, 'unpool')
        self.method = method
        self.LN = nn.LayerNorm([input_channel, self.output_size]) #Todo
        #self.BN = nn.BatchNorm1d(input_channel)
    def forward(self, tensor):
        #B T N C
        assert tensor.dim() == 4  # batch_size * input_channel * v_num
        B, T, N, C = tensor.shape
        tensor = torch.reshape(tensor, (B*T, N, C))  # B V I
        tensor = tensor.permute(0, 2, 1) # B I V
        output = torch.zeros((tensor.shape[0], tensor.shape[1], self.output_size), device=tensor.device)
        output[:,:,:self.input_size] = tensor
        output = self.LN(output)
        #output = self.BN(output)
        output = torch.reshape(output, (B, T, C, -1 ))
        output = output.permute(0, 1, 3, 2) # B T N C

        return output


class SparseSphereConv(nn.Module):
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=9,
                 stride=1,
                 select_index=None,
                 pre_layers=None,
                 post_layers=None,
                 *agrs, **kwargs):
        super(SparseSphereConv, self).__init__()
        self.index, itp_mat = gen_indexes(graph_level, 'conv', kernel_size)
        self.register_buffer("itp_mat", itp_mat)  # v_num * 7 * kernel_size
        self.conv = nn.Conv2d(input_channel, output_channel, (1, kernel_size))
        self.graph_level = graph_level
        self.kernel_size = kernel_size
        self.stride = stride
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        # select_index1: 根据输入选择
        if select_index is None:
            select_index1 = np.arange(v_num_dict[graph_level])
        else:
            select_index1 = np.array(select_index).squeeze()
            assert select_index1.ndim == 1
        # select_index2: 根据stride选择
        select_index2 = np.arange(v_num_dict[graph_level-stride+1])
        self.select_index = np.intersect1d(select_index1, select_index2)
        assert len(self.select_index) > 0

    def sparse_prepare(self, tensor, bs):
        v_num = v_num_dict[self.graph_level]
        assert bs*v_num == tensor.shape[1]  # inc * (bs*v_num) * 7
        min = torch.min(tensor.detach(), 0).values  # (bs*v_num) * 7
        min = torch.min(min, 1).values  # (bs*v_num)
        min = np.where(np.array(min.cpu())!=0)
        max = torch.max(tensor.detach(), 0).values  # (bs*v_num) * 7
        max = torch.max(max, 1).values  # (bs*v_num)
        max = np.where(np.array(max.cpu())!=0)
        # select_index3: min 和 max 其中一个不是0，就要传入下一步计算
        select_index3 = np.union1d(min, max)
        # select_index12: 设置的 select, 但是原来只有一个batch, 要扩张一下
        sel_idx_base = (np.arange(bs) * v_num).repeat(len(self.select_index))
        select_index12 = np.tile(self.select_index, bs)
        select_index12 = select_index12 + sel_idx_base
        select_index = np.intersect1d(select_index3, select_index12)

        tensor = tensor[:, select_index, :]  # inc * (bs*v_num2) * 7
        if self.kernel_size == 1:
            return tensor, select_index, self.itp_mat

        itp_mat = self.itp_mat.repeat(bs, 1, 1)  # (bs*v_num) * 7 * kernel_size
        itp_mat = itp_mat[select_index]  # (bs*v_num2) * 7 * kernel_size
        return tensor, select_index, itp_mat

    def sparse_reset(self, tensor, select_index, bs):
        # outc * (bs*v_num2)
        out_shape = [tensor.shape[0], bs*v_num_dict[self.graph_level]]
        out_tensor = torch.zeros(out_shape, device=tensor.device)  # outc * (bs*v_num)
        out_tensor[:, select_index] = tensor
        out_tensor = out_tensor.reshape((tensor.shape[0], bs, -1))  # outc * bs * v_num
        v_num3 = v_num_dict[self.graph_level - self.stride + 1]
        out_tensor = out_tensor[:, :, :v_num3]  # outc * bs * v_num3
        return out_tensor.permute(1, 0, 2)  # bs * outc * v_num3

    def interpolate(self, tensor):
        bs = tensor.shape[0]  # batch_size * input_channel * v_num
        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num * 7 * bs * inc
        tensor = tensor.permute(3, 2, 0, 1)  # inc * bs * v_num * 7
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[3])  # inc * (bs*v_num) * 7
        tensor, select_idx, itp_mat = self.sparse_prepare(tensor, bs)  # inc * (bs*v_num2) * 7
        # inc * (bs*v_num2) * 7  # (bs*v_num2) * 7 * kernel_size
        tensor = tensor.unsqueeze(-2)  # inc * (bs*v_num2) * 1 * 7
        if self.kernel_size == 1:
            tensor = tensor[:, :, :, -1]  # inc * (bs*v_num2) * 1
            tensor = tensor.unsqueeze(0)  # 1 * inc * (bs*v_num2) * 1
            return tensor, select_idx, bs

        # out_tensor = Interploate.apply(tensor, itp_mat)  # inc * (bs*v_num2) * 1 * ks
        # 将上一行拆分计算。直接拆分inc这个通道即可
        out_shape = [tensor.shape[0], tensor.shape[1], 1, self.kernel_size]
        out_tensor = torch.zeros(out_shape, device=tensor.device)  # inc * (bs*v_num2) * 1 * ks
        mem_cost = out_shape[0] * out_shape[1] * out_shape[3] * 4 / 1024 / 1024
        mem_goal = 1000
        inc_break = int(mem_cost//mem_goal) + 1
        inc_pb = tensor.shape[0]//inc_break + 1  # input_channel per break
        for cpb in range(inc_break):
            d = cpb * inc_pb
            u = (cpb+1) * inc_pb
            out_tensor[d:u] = Interploate.apply(tensor[d:u], itp_mat)

        tensor = out_tensor.squeeze(-2).unsqueeze(0)  # 1 * inc * (bs*v_num2) * ks
        return tensor, select_idx, bs

    def forward(self, tensor):
        if self.pre_layers:
            tensor = self.pre_layers(tensor)
        assert tensor.dim() == 3  # batch_size * input_channel * v_num

        tensor, select_index, bs = self.interpolate(tensor)  # 1 * inc * (bs*v_num2) * ks
        tensor = self.conv(tensor)  # 1 * outc * (bs*v_num2) * 1
        tensor = tensor.squeeze(-1).squeeze(0)  # outc * (bs*v_num2)
        tensor = self.sparse_reset(tensor, select_index, bs)  # bs * outc * v_num3

        if self.post_layers:
            tensor = self.post_layers(tensor)
        return tensor


class SelectSphereConv(nn.Module):
    # 跟 SphereConv 相比，多了 stride 和输入 index 选择
    # 跟 SparseSphereConv 相比，少了自动判断0点不计算插值和卷积
    def __init__(self, graph_level,
                 input_channel,
                 output_channel,
                 kernel_size=9,
                 stride=1,
                 select_index=None,
                 pre_layers=None,
                 post_layers=None,
                 *agrs, **kwargs):
        super(SelectSphereConv, self).__init__()
        self.index, itp_mat = gen_indexes(graph_level, 'conv', kernel_size)
        self.register_buffer("itp_mat", itp_mat)  # v_num * 7 * kernel_size
        self.conv = nn.Conv2d(input_channel, output_channel, (1, kernel_size))
        self.graph_level = graph_level
        self.kernel_size = kernel_size
        self.stride = stride
        self.pre_layers = pre_layers
        self.post_layers = post_layers

        # select_index1: 根据输入选择
        if select_index is None:
            select_index1 = np.arange(v_num_dict[graph_level])
        else:
            select_index1 = np.array(select_index).squeeze()
            assert select_index1.ndim == 1
        # select_index2: 根据stride选择
        select_index2 = np.arange(v_num_dict[graph_level-stride+1])
        self.select_index = np.intersect1d(select_index1, select_index2)
        assert len(self.select_index) > 0
        if kernel_size > 1:
            self.itp_mat = self.itp_mat[self.select_index]

    def sparse_reset(self, tensor):
        # batch_size * outc * v_num2
        v_num3 = v_num_dict[self.graph_level - self.stride + 1]
        out_shape = [tensor.shape[0], tensor.shape[1], v_num3]
        out_tensor = torch.zeros(out_shape, device=tensor.device)  # batch_size * outc * v_num3
        out_tensor[:, :, self.select_index] = tensor
        return out_tensor

    def interpolate(self, tensor):
        tensor = tensor.permute(2, 0, 1)  # v_num * bs * inc
        tensor = tensor[self.index]  # v_num * 7 * bs * inc
        tensor = tensor[self.select_index]  # v_num2 * 7 * bs * inc
        tensor = tensor.permute(2, 3, 0, 1)  # bs * inc * v_num2 * 7

        tensor = tensor.unsqueeze(-2)  # bs * inc * v_num2 * 1 * 7
        if self.kernel_size == 1:
            tensor = tensor[:, :, :, :, -1]  # bs * inc * v_num2 * 1
            return tensor

        # self.itp_mat  # v_num2 * 7 * ks
        # out_tensor = torch.matmul(tensor, self.itp_mat)  # bs * inc * v_num * 1 * ks
        out_shape = [tensor.shape[0], tensor.shape[1], tensor.shape[2], 1, self.kernel_size]
        out_tensor = torch.zeros(out_shape, device=tensor.device)  # bs * inc * v_num2 * 1 * ks
        mem_cost = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[4] * 4 / 1024 / 1024
        mem_goal = 1000
        inc_break = int(mem_cost//mem_goal) + 1
        inc_pb = tensor.shape[0]//inc_break + 1  # input_channel per break
        for cpb in range(inc_break):
            d = cpb * inc_pb
            u = (cpb+1) * inc_pb
            out_tensor[:, d:u] = Interploate.apply(tensor[:, d:u], self.itp_mat)

        out_tensor = out_tensor.squeeze(-2)  # bs * inc * v_num2 * ks
        return out_tensor

    def forward(self, tensor):
        if self.pre_layers:
            tensor = self.pre_layers(tensor)
        assert tensor.dim() == 3  # batch_size * input_channel * v_num

        tensor = self.interpolate(tensor)  # batch_size * input_channel * v_num2 * kernel_size
        tensor = self.conv(tensor)  # batch_size * outc * v_num2 * 1
        tensor = tensor.squeeze(-1)  # batch_size * outc * v_num2
        tensor = self.sparse_reset(tensor)  # bs * outc * v_num3

        if self.post_layers:
            tensor = self.post_layers(tensor)
        return tensor


if __name__ == "__main__":
    # test_conv()
    gl, inc, ouc = 1, 192, 192
    conv = SelectSphereConv(gl, inc, ouc, kernel_size=1, stride=2, select_index=None)
    # pdb.set_trace()
    tensor = torch.randn(5, 192, 42)
    # indexes = 3*np.arange(3000)
    # tensor[:, :, indexes] = 0
    out = conv(tensor)
    pdb.set_trace()
