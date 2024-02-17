import torch
import torch.nn as nn
import numpy as np

import pdb, sys, os, time
file_path = os.path.dirname(os.path.abspath(__file__))
# file_name = os.path.join(file_path, 'graph_info.npy')
sys.path.append(os.path.split(file_path)[0])
from spasgcn.graph_basic import S_Graph, v_num_dict, timer
EPSILON = np.finfo('float').eps

def load_graph_info(graph_level):
    if graph_level > 4:
        file_name = os.path.join(file_path, f'graph_info_{graph_level}.npy')
    else:
        file_name = os.path.join(file_path, f'graph_info_4m.npy')
    try:
        graph_info = np.load(file_name, allow_pickle=True).item()
    except FileNotFoundError:
        graph_info = {}

    if f"house_{graph_level}" in graph_info.keys():
        house = graph_info[f'house_{graph_level}']
        neighbor = graph_info[f'neighbor_{graph_level}']
        angle = graph_info[f'angle_{graph_level}']
        neip = graph_info[f'neip_{graph_level}']
        wp = graph_info[f'wp_{graph_level}']
        return house, neighbor, angle, neip, wp

    print(f'Generating graph info of level {graph_level}')
    if graph_level > 4:
        house = S_Graph(graph_level, 'fa')
        graph_neighbor = np.zeros((house.v_num, 7))
        graph_angle = np.zeros((house.v_num, 7))
        graph_neip = np.zeros((house.v_num, 7, 3))
        graph_wp = np.zeros((house.v_num, 5, 3))
        for i in range(12):
            neighbor = house.vertices[i].neighboridxs + [i] + [i]
            angle = house.vertices[i].neighboragls + [100] + [100]
            neip = house.vertices[i].neip_xyz + [[i, i, i]] + [[i, i, i]]
            wp = house.vertices[i].wp_xyz + [[100, 100, 100]]
            graph_neighbor[i] = np.array(neighbor)
            graph_angle[i] = np.array(angle)
            graph_neip[i] = np.array(neip)
            graph_wp[i] = np.array(wp)
        for i in range(12, house.v_num):
            neighbor = house.vertices[i].neighboridxs + [i]
            angle = house.vertices[i].neighboragls + [100]
            neip = house.vertices[i].neip_xyz + [[i, i, i]] 
            wp = house.vertices[i].wp_xyz + [[100, 100, 100]] 
            graph_neighbor[i] = np.array(neighbor)
            graph_angle[i] = np.array(angle)
            graph_neip[i] = np.array(neip)
            graph_wp[i] = np.array(wp)
        graph_info[f'house_{graph_level}'] = house
        graph_info[f'neighbor_{graph_level}'] = graph_neighbor
        graph_info[f'angle_{graph_level}'] = graph_angle
        graph_info[f'neip_{graph_level}'] = graph_neip
        graph_info[f'wp_{graph_level}'] = graph_wp
        np.save(file_name, graph_info)
        return house, graph_neighbor, graph_angle, graph_neip, graph_wp

    for gl in range(graph_level+1):
        house = S_Graph(gl, 'fa')
        graph_neighbor = np.zeros((house.v_num, 7))
        graph_angle = np.zeros((house.v_num, 7))
        graph_neip = np.zeros((house.v_num, 7, 3))
        graph_wp = np.zeros((house.v_num, 5, 3))
        for i in range(12):
            neighbor = house.vertices[i].neighboridxs + [i] + [i]
            angle = house.vertices[i].neighboragls + [100] + [100]
            neip = house.vertices[i].neip_xyz + [[i, i, i]] + [[i, i, i]]
            wp = house.vertices[i].wp_xyz + [[100, 100, 100]] 
            graph_neighbor[i] = np.array(neighbor)
            graph_angle[i] = np.array(angle)
            graph_neip[i] = np.array(neip)
            graph_wp[i] = np.array(wp)
        for i in range(12, house.v_num):
            neighbor = house.vertices[i].neighboridxs + [i]
            angle = house.vertices[i].neighboragls + [100]
            neip = house.vertices[i].neip_xyz + [[i, i, i]] 
            wp = house.vertices[i].wp_xyz + [[100, 100, 100]]  
            graph_neighbor[i] = np.array(neighbor)
            graph_angle[i] = np.array(angle)
            graph_neip[i] = np.array(neip)
            graph_wp[i] = np.array(wp)
        graph_info[f'house_{gl}'] = house
        graph_info[f'neighbor_{gl}'] = graph_neighbor
        graph_info[f'angle_{gl}'] = graph_angle
        graph_info[f'neip_{gl}'] = graph_neip
        graph_info[f'wp_{gl}'] = graph_wp
    np.save(file_name, graph_info)
    house = graph_info[f'house_{graph_level}']
    neighbor = graph_info[f'neighbor_{graph_level}']
    angle = graph_info[f'angle_{graph_level}']
    neip = graph_info[f'neip_{graph_level}']
    wp = graph_info[f'wp_{graph_level}']
    return house, neighbor, angle, neip, wp


def interpolate_prepare(angle, kernel_size):
    print('recal wh method')
    if kernel_size == 1:  # 1 * 1 卷积
        return torch.zeros(1,1).cuda()
    angle = angle * (kernel_size-1)/2  # v_num * 7
    assert angle.equal(angle.abs())
    angle_base_f = (angle.int()%(kernel_size-1)).long()
    angle_base_b = ((angle_base_f+1)%(kernel_size-1)).long()
    angle_coff_b = angle%1
    angle_coff_f = 1 - angle_coff_b

    v_num = angle.shape[0]
    base = torch.arange(v_num) * kernel_size  # v_num
    base = base.cuda()
    itp_mat = torch.zeros(7, v_num, kernel_size).cuda()  # 7 * v_num * kernel_size

    for i in range(5):
        tmp = torch.zeros(v_num*kernel_size, dtype=torch.float).cuda()
        index_f = angle_base_f.T[i] + base  # v_num
        index_b = angle_base_b.T[i] + base
        tmp[index_f] = angle_coff_f.T[i]
        tmp[index_b] = angle_coff_b.T[i]
        itp_mat[i] = tmp.reshape((v_num, kernel_size))  # v_num * kernel_size
    i = 5  # 前12个点不存在这一步，后面的点有第六个邻居
    tmp = torch.zeros(v_num*kernel_size, dtype=torch.float).cuda()
    index_f = angle_base_f.T[i][12:] + base[12:]
    index_b = angle_base_b.T[i][12:] + base[12:]
    tmp[index_f] = angle_coff_f.T[i][12:]
    tmp[index_b] = angle_coff_b.T[i][12:]
    itp_mat[i] = tmp.reshape((v_num, kernel_size))  # v_num * kernel_size
    # itp_mat[:, :12, :] = 6/5 * itp_mat[:, :12, :]  # 归一化
    i = 6  # 加上自身
    itp_mat[i, :, -1] = 1  # 7 * v_num * kernel_size
    itp_mat[:, :12, :] = 7/6 * itp_mat[:, :12, :]  # 归一化

    return itp_mat.permute(1,0,2)  # v_num * 7 * kernel_size

def interpolate_prepare_nor(neip, wp, angle, kernel_size=5):
    print('recall snor method')
    if kernel_size == 1:  # 1 * 1 卷积
        return torch.zeros(1,1)
    angle = angle * (kernel_size-1)/2  # v_num * 7
    assert angle.equal(angle.abs())
    w_base_f = (angle.int()%(kernel_size-1)).long()  # v_num * 7
    w_base_b = ((w_base_f+1)%(kernel_size-1)).long()
    wp_f = w_base_f.unsqueeze(-1).expand(-1, -1, 3) # v_num * 7 * 3
    wp_b = w_base_b.unsqueeze(-1).expand(-1, -1, 3)
    wp_f = torch.gather(wp, 1, wp_f) # wp: v_num * 5 * 3 --- w_base_f: v_num * 7 * 3 ---->  wp_f: v_num * 7 * 3
    wp_b = torch.gather(wp, 1, wp_b)   
    #w_coff_f = torch.dot(wp_f, neip)  # wp_f: v_num * 7 * 3, neip: v_num * 7 *3 >> v_num * 7
    #w_coff_b = torch.dot(wp_b, neip)   
    w_coff_f = torch.sum(wp_f*neip, -1)
    w_coff_b = torch.sum(wp_b*neip, -1)
    #normalize weight
    #w_coff_fn = w_coff_f/(w_coff_f+w_coff_b+EPSILON)
    #w_coff_bn = w_coff_b/(w_coff_f+w_coff_b+EPSILON)
    w_coff_fn = torch.exp(w_coff_f)/(torch.exp(w_coff_f)+torch.exp(w_coff_b)+EPSILON) #softmax 20220210
    w_coff_bn = torch.exp(w_coff_b)/(torch.exp(w_coff_f)+torch.exp(w_coff_b)+EPSILON)

    v_num = angle.shape[0]
    base = torch.arange(v_num) * kernel_size  # v_num
    itp_mat = torch.zeros(7, v_num, kernel_size) # 7 * v_num * 5    
    
    for i in range(5):
        tmp = torch.zeros(v_num*kernel_size, dtype=torch.float)
        index_f = w_base_f.T[i] + base  # v_num
        index_b = w_base_b.T[i] + base
        tmp[index_f] = w_coff_fn.T[i]
        tmp[index_b] = w_coff_bn.T[i]
        itp_mat[i] = tmp.reshape((v_num, kernel_size))  # v_num * kernel_size
    i = 5  # 前12个点不存在这一步，后面的点有第六个邻居
    tmp = torch.zeros(v_num*kernel_size, dtype=torch.float)
    index_f = w_base_f.T[i][12:] + base[12:]
    index_b = w_base_b.T[i][12:] + base[12:]
    tmp[index_f] = w_coff_fn.T[i][12:]
    tmp[index_b] = w_coff_bn.T[i][12:]
    itp_mat[i] = tmp.reshape((v_num, kernel_size))  # v_num * kernel_size
    # itp_mat[:, :12, :] = 6/5 * itp_mat[:, :12, :]  # 归一化
    i = 6  # 加上自身
    itp_mat[i, :, -1] = 1  # 7 * v_num * kernel_size
    itp_mat[:, :12, :] = 7/6 * itp_mat[:, :12, :]  # 归一化   (可调)
    return itp_mat.permute(1,0,2)  # v_num * 7 * kernel_size



def gen_indexes(graph_level, client='conv', kernel_size=9, *args, **kwargs):
    assert client in ['conv', 'pool', 'unpool']
    _, neighbor, angle, neip, wp = load_graph_info(graph_level)  # v_num * 7

    if client == 'conv':        
        conv_method = kwargs['conv_method']
        if conv_method == 'wh':
            index = torch.tensor(neighbor, dtype=torch.long).cuda()
            angle = torch.tensor(angle, dtype=torch.float).cuda()
            itp_mat = interpolate_prepare(angle, kernel_size)
        elif conv_method == 'nor':
            index = torch.tensor(neighbor, dtype=torch.long)
            angle = torch.tensor(angle, dtype=torch.float)
            neip = torch.tensor(neip, dtype=torch.float)
            wp = torch.tensor(wp, dtype=torch.float)            
            itp_mat = interpolate_prepare_nor(neip, wp, angle, kernel_size)
        return index, itp_mat

    elif client == 'pool':
        v_num_prime = v_num_dict[graph_level-1]
        index = torch.tensor(neighbor[:v_num_prime], dtype=torch.long).cuda()
        return index

    elif client == 'unpool':
        v_num = v_num_dict[graph_level]
        v_num_prime = v_num_dict[graph_level+1]
        return v_num, v_num_prime

def mkdir(dir):
    root = os.path.split(dir)[0]
    try:
        os.mkdir(root)
    except:
        pass
    try:
        os.mkdir(dir)
    except:
        pass

def show_memory(loc=None):
        print('Memory status: ', end='')
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB, ", end='')
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB, ", end='')
        print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB, ", end='')
        print(f"Max Reserved: {torch.cuda.max_memory_reserved()/1024**2:.1f}MB", end='')
        if loc:
            print(f', at: {loc}')
        else:
            print()


if __name__ == "__main__":
    gl, ks = 1, 9
    index, itp_mat = gen_indexes(gl, ks, client='conv')
