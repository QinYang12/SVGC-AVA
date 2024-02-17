import glob
import os
import pickle
import pdb
import cv2
import numpy as np
from spasgcn.utils import load_graph_info

graph_num=21
seconds = 30
len_snippet= 1
rate = 5
label_rate = 5


def static():
    #static frame only in vid_train
    image_mean = np.zeros(3)
    image_idx = 0
    vid_file = vid_train
    for vid in vid_file:
        vid_root = "/home/yq/Audio/Audio_visual/data/Qin/frame/%s"%(vid)
        for frame in os.listdir(vid_root):
            img = cv2.imread(os.path.join(vid_root, frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_mean += np.mean(img, (0, 1)) # mean of R,G,B
            image_idx +=1
        print(vid, image_idx)
    image_mean = image_mean/image_idx

    image_std = np.zeros(3)
    image_index = 0
    for vid in vid_file:
        vid_root = "/home/yq/Audio/Audio_visual/data/Qin/frame/%s"%(vid)
        for frame in os.listdir(vid_root):
            img = cv2.imread(os.path.join(vid_root, frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            image_std += np.sum((img-image_mean)**2, (0, 1))
            image_idx +=1
        print(vid, image_idx)
        print(H, W)
    image_std = np.sqrt(image_std/(image_idx*H*W))

    return image_mean, image_std    # RGB
    #visual ame, label


def generate(act):
    house, _ , _, _, _ = load_graph_info(5)
    data = {}
    #imgs = np.zeros((graph_num*seconds, len_snippet, house.v_num, 3))
    #aems = np.zeros((graph_num*seconds, len_snippet, house.v_num))
    #labels = np.zeros((graph_num*seconds, house.v_num))
    labels = []
    imgs = []
    aems = []
    index_l = []
    index_f = []

    if act == 'train':
        vid_file = vid_train
    elif act == 'test':
        vid_file = vid_test
    
    for vid in vid_file:
        print(vid)
        sal_root = "/home/yq/Audio/Audio_visual/data/Qin/saliency_small/%s"%(vid)
        frame_root = "/home/yq/Audio/Audio_visual/data/Qin/frame/%s"%(vid)
        aem_root = "/home/yq/Audio/Audio_visual/data/Qin/AEM/%s"%(vid)

        for j in range(int(len_snippet/2)*rate, len(os.listdir(sal_root))-int(len_snippet/2)*rate, label_rate):
            salmap = cv2.imread(os.path.join(sal_root, f'{j}.png'))
            print(f'{j}.png')
            salmap = cv2.cvtColor(salmap, cv2.COLOR_BGR2GRAY)
            salmap = house.Node_correspondence(salmap)
            labels.append(salmap)
            index_l.append(j)
            
            img = []
            aemlen = []
            for k in range(len_snippet):
                frame = cv2.imread(os.path.join(frame_root, f'frame_{j-int(len_snippet/2)*rate+k*rate}.png'))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = (frame-mean)/std #RGB
                frame = house.Node_correspondence(frame)
                img.append(frame)

                aem = cv2.imread(os.path.join(aem_root, f'aem_{j-int(len_snippet/2)*rate+k*rate}.png'))
                aem = cv2.cvtColor(aem, cv2.COLOR_BGR2GRAY)
                aem = house.Node_correspondence(aem)
                aemlen.append(aem)
                print(f'{j-int(len_snippet/2)*rate+k*rate}')
            imgs.append(img)
            aems.append(aemlen)
            index_f.append(j-int(len_snippet/2)*rate+k*rate)

    data['label'] = np.array(labels, dtype=np.float32) #(real:sum=1)0~255
    data['img'] = np.array(imgs, dtype=np.float32)     #mean-std
    data['aem'] = np.array(aems, dtype=np.float32)/255 #(real:0-1.18xx)0~1 
    data['index_l'] = np.array(index_l, dtype=np.int)
    data['index_f'] = np.array(index_f, dtype=np.int)
    print(data['img'].shape)
    print(data['label'].shape)
    print(data['aem'].shape)
    datapath = './data/Qin/'+f'Qin_{act}_G5_L{len_snippet}_R{rate}_small_random'
    f_save = open(datapath+'.pkl', 'wb')
    pickle.dump(data, f_save, protocol=4)
    f_save.close()

    # f_read = open(datapath+'.pkl', 'rb')
    # data = pickle.load(f_read)
    # f_read.close()


if __name__  == '__main__':
    # todo:怎么区别chao的train和test(7:3)/(8:2)
    # PVS:61:15
    # chao:15:6 / 17:4
    # qin: 42-18 / 48:12
    #return frame:mean_std normalize; label:(real:sum=1)0-255, aem:(real:0-1.18xx)0-255
    
    #vid_train = ['L31', 'L34', 'L09', 'L33', 'L18', 'L24', 'L52', 'L47', 'L49', 'L59', 'L43', 'L06','L38', 'L45', 'L39', 'L56', 'L54', 'L10', 'L53', 'L21', 'L36', 'L25', 'L12', 'L20', 'L35', 'L04', 'L22', 'L17', 'L55', 'L02', 'L28','L08', 'L16', 'L07', 'L58', 'L01', 'L14', 'L19', 'L48', 'L05', 'L13','L46', 'L23', 'L27', 'L26', 'L60', 'L51', 'L40'] #48
    #vid_test = ['L03', 'L11', 'L15', 'L29','L30', 'L32', 'L37', 'L41', 'L42', 'L44', 'L50', 'L57'] #12
    #vid_train = ['L34', 'L03', 'L09', 'L33', 'L50', 'L18', 'L24', 'L52', 'L47', 'L49', 'L59', 'L43', 'L06', 'L38', 'L45', 'L44', 'L39', 'L29', 'L56', 'L54', 'L10', 'L53', 'L21', 'L36', 'L25', 'L12', 'L31', 'L20', 'L11', 'L32', 'L35', 'L04', 'L22', 'L37', 'L17', 'L55', 'L02', 'L30', 'L28', 'L08', 'L16', 'L07', 'L58', 'L42', 'L01', 'L14', 'L19', 'L57'] # random split
    #vid_test = ['L48', 'L41', 'L05', 'L13', 'L46', 'L23', 'L27', 'L26', 'L60', 'L15', 'L51', 'L40'] # random split
              
    vid_train = ['L34', 'L09', 'L33', 'L50', 'L18', 'L24', 'L52', 'L47', 'L49', 'L59', 'L43', 'L06', 'L38', 'L45', 'L44', 'L39', 'L29', 'L56', 'L54', 'L10', 'L53', 'L21', 'L36', 'L25', 'L12', 'L31', 'L20', 'L11', 'L35', 'L04', 'L17', 'L02', 'L28', 'L08', 'L16', 'L07', 'L58', 'L42', 'L01', 'L14', 'L57', 'L48', 'L05', 'L46', 'L60', 'L40' ] # 46 fix+random           
    vid_test =  ['L03', 'L15', 'L30', 'L55', 'L32', 'L37', 'L41', 'L13', 'L23', 'L26', 'L51'] # 11 fix+random

    #mean, std = static()
    #pdb.set_trace()
    #mean = [107.9473805 , 104.58465794, 103.53123063]
    #std = [46.72126134, 47.57749019, 50.07759837]
    #mean = [110.28996499, 108.26372459, 107.27699197] # random split
    #std = [45.82535692, 46.96157094, 50.37198691] # random split
    mean = [111.18318266, 107.75611663, 105.34116761]
    std = [46.76133558, 47.92816476, 51.48270564]
    generate('train')
    generate('test')
