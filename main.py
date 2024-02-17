import spasgcn as spag
from spasgcn.utils import timer, v_num_dict, mkdir
from spasgcn.test_metric_chao import metric_chao
from spasgcn.test_metric_qin import metric_qin

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import numpy as np

import argparse
import sys, pdb, os, time, datetime, json
file_path = os.path.dirname(os.path.abspath(__file__))


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# setting parameters
parser = argparse.ArgumentParser()
#data
parser.add_argument('-dph', '--dataset_path', default='data', type=str)
parser.add_argument('-nw', '--num_workers', default=8, type=int)
parser.add_argument('-ds', '--data_set', default='Qin', choices=['Chao', 'Qin'])
parser.add_argument('-gaussian', '--gaussian', default='small', choices=['small', 'large']) #large:15, small:7
# model
parser.add_argument('-ms', '--model', default='model0', choices=['model0','model1','model2', 'model21', 'model22', 'model3' ,'model4','model5', 'modelR', 'model2c', 'model2a', 'model2w', 'model2Sp', 'model2HR', 'model21H'])
parser.add_argument('-gl', '--graph_level', default=5, type=int)
parser.add_argument('-ks', '--kernel_s', default=5, type=int)
parser.add_argument('-kt', '--kernel_t', default=3, type=int)
parser.add_argument('-pt', '--pool_t', default=4, type=int)#2
parser.add_argument('-convm', '--conv_method', default='nor', choices=['wh', 'nor'])
parser.add_argument('-ch', '--mchannel', default=32, type=int)
parser.add_argument('-nbl', '--Nblock', default=4, type=int)
parser.add_argument('-Tloc', '--Tlocation', default=[3,4,5], type=list)
parser.add_argument('-p', '--pooling_method', default='mean', choices=['sample', 'max', 'mean'])
parser.add_argument('-bs', '--batch_size', default=32, type=int)
parser.add_argument('-Tframe', '--len_snippet', default=1, type=int)#5
parser.add_argument('-rate', '--rate', default=5, type=int)#5
parser.add_argument('-label_rate', '--label_rate', default=5, type=int)#5
# train
parser.add_argument('-losrd', '--loss_reduce', default='sum', choices=['mean','sum'])
parser.add_argument('-lostp', '--loss_type', default='KL', choices=['KL','CE', 'MSE'])
parser.add_argument('-gpu', '--gpu_num', type=str)
parser.add_argument('-opt', '--optimizer_name', default='adam', choices=['adam', 'sgd'])
parser.add_argument('-weight_decay', '--weight_decay', default=0.0001, type=float) 
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float)
parser.add_argument('-momm', '--momentum', default=0.9, type=float)
parser.add_argument('-en', '--epoch_num', default=100, type=int)
parser.add_argument('-tri', '--train_interval', default=10, type=int)
# actions
parser.add_argument('-act', '--action', default='train', choices=['train', 'test', 'metric'])
parser.add_argument('-goal', '--loss_goal', default=1.5, type=float)
parser.add_argument('-load', '--load_model')
parser.add_argument('--load_args', action="store_true")
parser.add_argument('-save', '--save_path')
parser.add_argument('-showout', '--showout', default='log', choices=['log', 'screen'])
args = parser.parse_args()

if args.gpu_num in ['no', 'cpu', '']:
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    use_gpu = False
elif args.gpu_num:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    use_gpu = True
else:
    use_gpu = torch.cuda.is_available()

if args.save_path:
    save_path = os.path.join('save', args.save_path)
else:
    now = datetime.datetime.now()
    save_path = os.path.join('save', now.strftime('%Y%m%d_%H%M%S')[2:])
writer = SummaryWriter(os.path.join(save_path, 'showloss'))
mkdir(save_path)

# save logs and commandline
file_write = open(os.path.join(save_path, 'logs.txt'), 'a')
if args.save_path and args.save_path != 'no':
    if args.showout == 'log':
        sys.stdout = file_write
    save_cmdline_args = os.path.join(save_path, 'commandline_args.txt')
    with open(save_cmdline_args, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

if args.load_args and args.load_model[-4:] != '.pth':
    load_cmdline_args = os.path.join('save', args.load_model, 'commandline_args.txt')
else:
    load_cmdline_args = 'commandline_args.txt'

try:
    with open(load_cmdline_args, 'r') as f:
        args.__dict__ = json.load(f)
except:
    pass

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------')

# load model
finchannel = 3
oinchannel = 2
ainchannel = 1
 
print('Building Model ...')
net = spag.Model_select(args.model, args.graph_level, args.kernel_s, args.kernel_t, pool_t=args.pool_t, finchannel=finchannel,ainchannel=ainchannel, mchannel=args.mchannel, Nblock=args.Nblock, conv_method=args.conv_method)

if args.load_model:
    if args.load_model[-4:] != '.pth':
        load_path = os.path.join('save', args.load_model, 'small.pth')
    else:
        load_path = args.load_model
    print('model load path is %s' % load_path)
    net.load_state_dict(torch.load(load_path, map_location='cpu'))    

if use_gpu:
    net = nn.DataParallel(net).cuda()
    print(net)
    print('device: cuda:', os.environ["CUDA_VISIBLE_DEVICES"])
else:
    print(net)
    print('device: cpu')
if args.optimizer_name == 'adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
elif args.optimizer_name == 'sgd':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=True, cooldown=5, min_lr=args.learning_rate/100)
# eval
@timer
def evalauate(data_loader_test):
    loss_ave, loss_num = 0, 0
    with torch.no_grad():
        for i_batch, (frame, label, aem, index_l, index_f) in enumerate(data_loader_test):
            if use_gpu:
                frame = frame.cuda()  # bs * inc * v_num
                label = label.cuda()
                aem = aem.cuda()
                Tlocation = torch.tensor(args.Tlocation).cuda()
            assert frame.shape[2] == v_num_dict[args.graph_level]
            net.eval()
            output = net(frame,aem, Tlocation)
            label = normalize(label, method='sum')
            #output = normalize(output, method='sum')
            if label.shape[0]==1:
                label = label.squeeze()
            if output.shape[0]==1:
                    output = output.squeeze()
            if args.loss_reduce == 'mean':
                if args.loss_type == 'MSE':
                    loss = nn.MSELoss(reduction='mean')(output, label)
                elif args.loss_type == 'CE':
                    loss = nn.CrossEntropyLoss(reduction='mean')(output.long(), label.long())
                elif args.loss_type == 'KL':
                    loss = nn.KLDivLoss(reduction='batchmean')(output.log(), label)
            elif args.loss_reduce == 'sum':
                if args.loss_type == 'MSE':
                    loss = nn.MSELoss(reduction='sum')(output, label)
                elif args.loss_type == 'CE':
                    loss = nn.CrossEntropyLoss(reduction='sum')(output, label.long())
                elif args.loss_type == 'KL':
                    loss = nn.KLDivLoss(reduction='sum')(output.log(), label)
            loss_ave += loss.detach()
            loss_num +=1

        loss_ave = float(loss_ave / loss_num)
        writer.add_scalar('test_loss', loss_ave, global_step)
        # writer.add_scalar('test_accuracy', accuracy, global_step)

        print(f"step {global_step}, tested {loss_num*args.batch_size} imgs, loss = {loss_ave:.6f}")
    return loss_ave

def test(data_set, act, final_pth):
    #load final model
    data_test = spag.Data_select(data_set, dataset_path, args.len_snippet, act, args.graph_level, args.gaussian)
    data_loader_test = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"found {len(data_test)} test images")
    file_write.flush()
    save_pth_name = final_pth.split('.')[0]
    pre_graph_path = os.path.join(save_path, f'{data_set}_{save_pth_name}_{act}_presal.npy')

    load_path = os.path.join(save_path, final_pth)
    print('model load path is %s' % load_path)
    net = spag.Model_select(args.model, args.graph_level, args.kernel_s, args.kernel_t, pool_t=args.pool_t, finchannel=finchannel,ainchannel=ainchannel, mchannel=args.mchannel, Nblock=args.Nblock, conv_method=args.conv_method)
    net.load_state_dict(torch.load(load_path, map_location='cpu'))
    if use_gpu:
        net = nn.DataParallel(net).cuda()
    
    #save result 
    predict = []
    with torch.no_grad():
        for i_batch, (frame, label, aem, index_l, index_f) in enumerate(data_loader_test):
            if use_gpu:
                frame = frame.cuda()  # bs * inc * v_num
                label = label.cuda()
                aem = aem.cuda()
                Tlocation = torch.tensor(args.Tlocation).cuda()
            assert frame.shape[2] == v_num_dict[args.graph_level]
            net.eval()
            output = net(frame,aem, Tlocation)
            predict.append(output.cpu().numpy())
        predict = np.array(predict).reshape(-1, v_num_dict[args.graph_level])
        np.save(pre_graph_path, predict)
        print('finish predict the saliency map')

def normalize(x, method='standard'):
    #todo
    if method == 'standard':
        mu = torch.mean(x, -1, keepdim=True)
        sigma = torch.std(x, -1, keepdim=True)
        res = (x-mu)/torch.sqrt(sigma)
    elif method == 'range':
        mat_min = torch.min(x, -1, keepdim=True)
        mat_max = torch.max(x, -1, keepdim=True)
        res = (x - mat_min) / (mat_max - mat_min)
    elif method == 'sum':
        mat_sum = torch.sum(x, -1, keepdim=True)
        res = x / mat_sum
    else:
        raise ValueError('method not in {"standard", "range", "sum"}')
    return res

if __name__ == '__main__':
    if args.dataset_path[:5] == file_path[:5]:
        dataset_path = args.dataset_path
    else:
        dataset_path = os.path.join(file_path, args.dataset_path)
    if args.action == 'test':
        test('Qin', 'train', 'small.pth')
        test('Qin', 'test', 'small.pth')
        test('Chao', 'test', 'small.pth')
        sys.exit()
    if args.action == 'metric':
        KL0, NSS0, CC0, ALL0 = metric_qin('Qin', 'small.pth', save_path, 'train', args.gaussian, args.len_snippet, args.rate, args.label_rate) 
        KL1, NSS1, CC1, ALL1 = metric_qin('Qin', 'small.pth', save_path, 'test', args.gaussian, args.len_snippet, args.rate, args.label_rate) 
        KL2, NSS2, CC2, ALL2 = metric_chao('Chao', 'small.pth', save_path, 'test') 

    # train and test
    # load data
    data_train = spag.Data_select(args.data_set, dataset_path, args.len_snippet, 'train', args.graph_level, args.gaussian)
    data_test = spag.Data_select(args.data_set, dataset_path,args.len_snippet, 'test', args.graph_level, args.gaussian)
    data_loader_train = DataLoader(data_train, args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_test = DataLoader(data_test, 1, shuffle=False, num_workers=args.num_workers)
    print(f"found {len(data_train)} train images and {len(data_test)} test images")
    
    #start train
    print("Start training")
    file_write.flush()
    min_loss, coverage_num, global_step = args.loss_goal, 0, 0

    for ep in range(args.epoch_num):
        loss_ave, loss_num, step_loss = 0, 0, 0
        print(f"\nEPOCH {ep + 1}", datetime.datetime.now())
        print(optimizer)
        for i_batch, (frame, label, aem, index_l, index_f) in enumerate(data_loader_train):
            #print(f'train {i_batch}  index{index_l}, {index_f}')
            if global_step % args.train_interval == 1:
                 time_start = time.time()
            time_start = time.time()
            if use_gpu:
                frame = frame.cuda()  # bs * inc * v_num
                label = label.cuda()
                aem = aem.cuda()
                Tlocation = torch.tensor(args.Tlocation).cuda()
            assert frame.shape[2] == v_num_dict[args.graph_level]
            net.train()
            output = net(frame,aem, Tlocation)
            label = normalize(label, method='sum')
            
            if label.shape[0]==1:
                label = label.squeeze()
            if output.shape[0]==1:
                output = output.squeeze()
            if args.loss_reduce == 'mean':
                if args.loss_type == 'MSE':
                    loss = nn.MSELoss(reduction='mean')(output, label)
                elif args.loss_type == 'CE':
                    loss = nn.CrossEntropyLoss(reduction='mean')(output, label.long())
                elif args.loss_type == 'KL':
                    loss = nn.KLDivLoss(reduction='batchmean')(output.log(), label)
            elif args.loss_reduce == 'sum':
                if args.loss_type == 'MSE':
                    loss = nn.MSELoss(reduction='sum')(output, label)
                elif args.loss_type == 'CE':
                    loss = nn.CrossEntropyLoss(reduction='sum')(output, label.long())
                elif args.loss_type == 'KL':
                    loss = nn.KLDivLoss(reduction='sum')(output.log(), label)
            loss_ave += loss.detach()
            step_loss += loss.detach()
            loss_num += 1
            writer.add_scalar('train_loss', loss, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            del frame, label, output, loss
            if global_step % args.train_interval == 0:
                duration = time.time() - time_start
                step_loss /= args.train_interval
                print(f"step {global_step-args.train_interval+1}-{global_step}, ", end='')
                print(f"train loss = {float(step_loss):.4f}, time = {duration:.2f}s")
                step_loss = 0

        test_loss = evalauate(data_loader_test)
        scheduler.step(test_loss)
        if test_loss < min_loss and args.save_path != 'no':
            coverage_num = 0
            min_loss = test_loss
            save_name = os.path.join(save_path, 'small.pth')            
            print('save model, test loss = %.2f, step = %d, save_path is: %s'% (test_loss, global_step, save_name))
            if use_gpu:
                torch.save(net.module.state_dict(), save_name)
            else:
                torch.save(net.state_dict(), save_name)
        elif test_loss >= min_loss and coverage_num <= 5: 
            coverage_num += 1
            print(f'the epoch num without converage is {coverage_num}')
        elif test_loss >= min_loss and coverage_num > 5: #early stop
            save_name = os.path.join(save_path, 'final.pth')            
            print('save model, test loss = %.2f, step = %d, save_path is: %s'% (test_loss, global_step, save_name))
            if use_gpu:
                torch.save(net.module.state_dict(), save_name)
            else:
                torch.save(net.state_dict(), save_name)
            break

        loss_ave = float(loss_ave / loss_num)
        writer.add_scalar('train_ep_loss', loss_ave, global_step)
        writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step)
        # writer.add_scalar('train_ep_accuracy', accuracy, global_step)
        print(f'EP: {ep+1}, train loss = {loss_ave:.6f}')
        print("final test loss till this epoch: %.2f" % (min_loss))
        file_write.flush()
        if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-6:
            break
    

    #generate the final saliency map and test metric
    final_metric = []
    for save_pth in ['small.pth', 'final.pth']:
        test('Qin', 'train', save_pth)
        test('Qin', 'test', save_pth)
        test('Chao', 'test', save_pth)

        KL0, NSS0, CC0, ALL0 = metric_qin('Qin', save_pth, save_path, 'train', args.gaussian, args.len_snippet, args.rate, args.label_rate) 
        KL1, NSS1, CC1, ALL1 = metric_qin('Qin', save_pth, save_path, 'test', args.gaussian, args.len_snippet, args.rate, args.label_rate) 
        KL2, NSS2, CC2, ALL2 = metric_chao('Chao', save_pth, save_path, 'test') 
        final_metric.append([[KL0, NSS0, CC0, ALL0], [KL1, NSS1, CC1, ALL1], [KL2, NSS2, CC2, ALL2]]) 
        print('final result for small_loss and final:')
        print(final_metric)
    file_write.close()