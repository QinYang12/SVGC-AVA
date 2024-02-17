import glob
import os
import pickle
import pdb
import cv2
import numpy as np
from spasgcn.utils import load_graph_info

graph_num=21
seconds = 30
len_snippet=1
rate = 5 #len_snippet=5--> rate =1

def static():
    #static frame only in vid_train
    image_mean = np.zeros(3)
    image_idx = 0
    for vid in vid_train:
        vid_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/frame", vid)
        for frame in os.listdir(vid_root):
            img = cv2.imread(os.path.join(vid_root, frame))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_mean += np.mean(img, (0, 1)) # mean of R,G,B
            image_idx +=1
        print(vid, image_idx)
    image_mean = image_mean/image_idx

    image_std = np.zeros(3)
    image_index = 0
    for vid in vid_train:
        vid_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/frame", vid)
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

    if act == 'train':
        vid_file = vid_train
    elif act == 'test':
        vid_file = vid_test
    
    for vid in vid_file:
        print(vid)
        sal_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/saliency", vid) 
        frame_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/frame", vid)
        aem_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/AEM", vid)

        for j in range(len(os.listdir(sal_root))):
            salmap = cv2.imread(os.path.join(sal_root, f'salmap_f_{j}.png'))
            print(j)
            salmap = cv2.cvtColor(salmap, cv2.COLOR_BGR2GRAY)
            salmap = house.Node_correspondence(salmap)
            labels.append(salmap)
            
            img = []
            aemlen = []
            for k in range(len_snippet):
                frame = cv2.imread(os.path.join(frame_root, f'frame_{j*len_snippet*rate+k*rate} .png'))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = (frame-mean)/std #RGB
                frame = house.Node_correspondence(frame)
                img.append(frame)

                aem = cv2.imread(os.path.join(aem_root, f'aem_{j*len_snippet*rate+k*rate} .png'))
                aem = cv2.cvtColor(aem, cv2.COLOR_BGR2GRAY)
                aem = house.Node_correspondence(aem)
                print(j*len_snippet*rate+k*rate)
                aemlen.append(aem)
            imgs.append(img)
            aems.append(aemlen)

    data['label'] = np.array(labels, dtype=np.float32)
    data['img'] = np.array(imgs, dtype=np.float32)
    data['aem'] = np.array(aems, dtype=np.float32)/255
    print(data['img'].shape)
    print(data['label'].shape)
    print(data['aem'].shape)
    datapath = './data/Chao/'+f'ttttt_{act}_G5_L{len_snippet}'
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
    #vid_train = ['ByBF08H-wDA', 'dpfkpZzZvqw', '6QUCaLvQ_3I', 'dd39herpgXA', 'kZB3KMhqqyI', '8feS1rNYEbg', 'oegasz59U7I', '5h95uTtPeck', 'Ngj6C_RMK1g', 'Bvu9m__ZX60', '8ESEI0bqrJ4', 'fryDy9YcbI4', 'MzcdEI-tSUc', 'idLVnagjl_s', 'nZJGt3ZVg3g'] #chao split 15
    #vid_test = ['Oue_XEKHq3g', '1An41lDIJ6Q', 'ey9J7w98wlI', 'OZOaN_5ymrc', 'RbgxpagCY_c_2', 'gSueCRQO_5g'] #chao split 6
    vid_test = ['idLVnagjl_s', 'ey9J7w98wlI', 'kZB3KMhqqyI', 'MzcdEI-tSUc', '8ESEI0bqrJ4','1An41lDIJ6Q','6QUCaLvQ_3I', '8feS1rNYEbg','ByBF08H-wDA','fryDy9YcbI4', 'RbgxpagCY_c_2','dd39herpgXA'] 
    #mean, std = static()
    #pdb.set_trace()
    #mean = [117.97154841, 108.71887277, 102.00159405] # chao split
    #std = [50.21038083, 48.9973199 , 50.76477383] # chao split
    mean = [111.18318266, 107.75611663, 105.34116761]
    std = [46.76133558, 47.92816476, 51.48270564]
    #generate('train')
    generate('test')