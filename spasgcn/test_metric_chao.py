from functools import partial
import numpy as np
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re, os, glob
import pdb
import math
import random
from spasgcn.utils import load_graph_info
from time import time
EPSILON = np.finfo('float').eps
class MyGaussianBlur():
    #初始化
    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema
    #高斯的计算公式
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
    #得到滤波模版
    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all
    #滤波函数
    def filter(self, image, template):
        #arr=np.array(image)
        height=image.shape[0]
        width=image.shape[1]
        newData=np.zeros((height, width))
        for i in range(self.radius, height-self.radius):
            for j in range(self.radius, width-self.radius):
                t=image[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1]
                a= np.multiply(t, template)
                newData[i, j] = a.sum()
        #newImage = Image.fromarray(newData)
        return newData

def FCB(): #for XuMai
    x = np.arange(0, 360, 1)
    y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(x, y)
    fcb = np.exp(-((Y-90)**2+(X-180)**2)/21.1**2)
    return fcb

def normalize(x, method='standard', axis=None):
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image,nbins)  # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)


def KLD(q, p):
    p = normalize(p, method='sum') #label
    q = normalize(q, method='sum') #predicted
    return np.sum(np.where(p != 0, p * np.log((p + EPSILON) / (q + EPSILON)), 0))

def AUC_Judd(saliency_map, fixation_map, jitter=False):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map

    if rand_sampler is None:
        r = np.random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r]  # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:, rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds) + 2)
        fp = np.zeros(len(thresholds) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k + 1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k + 1] = np.sum(S_rand[:, rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)  # Average across random splits

def AUC_shuffled(saliency_map, fixation_map, other_map, n_split=100, step_size=.1):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='constant')
    saliency_map = normalize(saliency_map, method='range')
    S = saliency_map.ravel()
    F = fixation_map.ravel()
    Oth = other_map.ravel()
    Sth = S[F]
    Nfixations = len(Sth)

    ind = np.where(Oth>0)[0]
    Nfixations_oth = np.min([Nfixations, len(ind)])
    randfix =  np.zeros((Nfixations_oth, n_split)) * np.nan
    for i in range(n_split):
        randind = ind[np.random.permutation(len(ind))]
        randfix[:, i] = S[randind[:Nfixations_oth]]

    auc = np.zeros((n_split))*np.nan
    for s in range(n_split):
        allthreshes = np.r_[0:np.max(np.r_[Sth, randfix[:, s]]):step_size][::-1]
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[0] = 0
        tp[-1] = 1
        fp[0] = 0
        fp[-1] = 1
        for k, thresh in enumerate(allthreshes):
            tp[k + 1] = np.sum(Sth >= thresh) / float(Nfixations)
            fp[k + 1] = np.sum(randfix[:, s] >= thresh) / float(Nfixations_oth)
        auc[s] = np.trapz(tp, fp)
    return np.mean(auc)

def AUC_shuffled_other(pred_sal, fix_map, base_map, n_split=100, step_size=.1):
    if pred_sal.shape != fix_map.shape:
        pred_sal = scipy.misc.imresize(pred_sal, fix_map.shape)
    assert(base_map.shape == fix_map.shape)
    pred_sal = pred_sal.flatten().astype(np.float)
    base_map = base_map.flatten().astype(np.float)
    fix_map = fix_map.flatten().astype(np.bool)
    pred_sal = (pred_sal - pred_sal.min()) / (pred_sal.max() - pred_sal.min())
    sal_fix = pred_sal[fix_map]
    sorted_sal_fix = np.sort(sal_fix)
    ind = np.where(base_map>0)[0]
    n_fix = sal_fix.shape[0]
    n_fix_oth = np.minimum(n_fix, ind.shape[0])

    rand_fix = np.zeros((n_fix_oth, n_split))
    for i in range(n_split):
        rand_ind = random.sample(ind.tolist(), n_fix_oth)
        rand_fix[:,i] = pred_sal[rand_ind]
    auc = np.zeros((n_split))
    for i in range(n_split):
        cur_fix = rand_fix[:, i]
        sorted_cur_fix = np.sort(cur_fix)
        max_val = np.maximum(cur_fix.max(), sal_fix.max())
        tmp_all_thres = np.arange(0, max_val, step_size)[::-1]
        tp = np.zeros((tmp_all_thres.shape[0]))
        fp = np.zeros((tmp_all_thres.shape[0]))
        for ind, thres in enumerate(tmp_all_thres):
            tp[ind] = (sorted_sal_fix.shape[0] - sorted_sal_fix.searchsorted(thres, side='left')) * 1. / n_fix
            fp[ind] = (sorted_cur_fix.shape[0] - sorted_cur_fix.searchsorted(thres, side='left')) * 1. / n_fix_oth
        tp = np.concatenate([[0], tp, [1]])
        fp = np.concatenate([[0], fp, [1]])
        auc[i] = np.trapz(tp, fp)
    return np.mean(auc)

def NSS(saliency_map, fixation_map):
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0, 1]


def SIM(saliency_map1, saliency_map2):
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3,
                      mode='constant')  # bi-cubic/nearest is what Matlab imresize() does by default
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def get_binsalmap_info(filename):
    get_binsalmap_infoRE = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
    name, width, height,Nframes, dtype = get_binsalmap_infoRE.findall(filename.split(os.sep)[-1])[0]
    width, height,Nframes, dtype = int(width), int(height),int(Nframes),int(dtype)
    return name, width, height, Nframes

def getSimVal(keys_order, metrics, salmap1, salmap2, fixmap1=None, fixmap2=None, other_map1=None, other_map2=None):
    #salmap1:predicted, fixmap1:predicted, other_map1:predicted
    #salmap2:label, fixmap2:label, other_map2:label
    values = []

    for metric in keys_order:

        func = metrics[metric][0]
        sim = metrics[metric][1]
        compType = metrics[metric][2]

        if not sim:
            if compType == "fix" and not "NoneType" in [type(fixmap1), type(fixmap2)]:
                m = func(salmap1, fixmap2)
            else:
                m = func(salmap1, salmap2)
        else:
            m = func(salmap1, fixmap2, other_map2)
        values.append(m)
    return values

def uniformSphereSampling(N):
    gr = (1 + np.sqrt(5)) / 2
    ga = 2 * np.pi * (1 - 1 / gr)

    ix = iy = np.arange(N)

    lat = np.arccos(1 - 2 * ix / (N - 1))
    lon = iy * ga
    lon %= 2 * np.pi

    return np.concatenate([lat[:, None], lon[:, None]], axis=1)

def unifs_generate(SAMPLING_TYPE, point_theta, height):
    if SAMPLING_TYPE == "Sin":
        unifS = None
        weight_ver = np.sin(np.linspace(0, np.pi, height))
    elif SAMPLING_TYPE.split("_")[0] == "GICOPix":
        unifS = np.array(point_theta)
        unifS[:, 0] = (0.5-unifS[:, 0])*(height-1)
        unifS[:, 1] = (unifS[:,1]+1)*0.5*(width-1)
        weight_ver = None
    elif SAMPLING_TYPE.split("_")[0] == "Sphere":
        print(int(SAMPLING_TYPE.split("_")[1]))
        unifS = uniformSphereSampling( int(SAMPLING_TYPE.split("_")[1]))
        unifS[:, 0] = unifS[:, 0] / np.pi * (height-1)
        unifS[:, 1] = unifS[:, 1] / (2*np.pi) * (width-1)
        unifS = unifS.astype(int)
        weight_ver = None
    return unifS, weight_ver

def ErpToSphere(unifS, erpmap, height):
    Spheremap = []
    W = 640
    H = 480
    for i in range(len(unifS)):
        x = unifS[i][1]
        y = unifS[i][0]
        x_int = int(x)
        y_int = int(y)
        dx = x-x_int
        dy = y-y_int
        pixel_bo_left = erpmap[min(y_int+1,H-1), x_int]
        pixel_bo_right = erpmap[min(y_int+1,H-1),min(x_int+1,W-1)]
        pixel_up_left = erpmap[y_int,x_int ]
        pixel_up_right = erpmap[y_int,min(x_int+1,W-1)]
        pixel_bo = (1-dx) * pixel_bo_left + (dx) * pixel_bo_right
        pixel_up = (1-dx) * pixel_up_left + (dx) * pixel_up_right
        pixel = (1-dy) * pixel_up + (dy) * pixel_bo
        Spheremap.append(pixel)
    return np.array(Spheremap)

def SphereToErp(final_height, sphere_img, point_theta):
    height = 72
    width=2*height
    unifs1 = np.array(point_theta)
    #scale : to be the same with the way in the Node_correspondence
    unifs1[:, 0] = (0.5-unifs1[:, 0])*(height-1) #-0.5~0.5 --> 
    unifs1[:, 1] = (unifs1[:,1]+1)*0.5*(width-1) #-1~1 -->

    if len(sphere_img.shape)==2:
        img=np.zeros((height, 2*height,sphere_img.shape[0]))
        sphere_img = np.transpose(sphere_img, [1,0])
    elif len(sphere_img.shape)==1:
        img=np.zeros((height, 2*height))
    for i in range(len(unifs1)):
        x = int(unifs1[i][1])
        y = int(unifs1[i][0])
        img[y,x] = sphere_img[i] #bilinear interpolation to do
    if len(sphere_img.shape)==2:
        img = np.transpose(img, [2,0,1])

    image = Image.fromarray(img)
    image = image.resize((640,480), Image.BICUBIC)
    #image = image.resize((2048,1024), Image.ANTIALIAS)
    img = np.array(image)
    #print('the negative number of Image.ANTIALIAS', (img<0).sum())
    img = normalize(img, method='range')

    '''
    #plt.figure()
    plt.imshow(img)
    plt.title('BICUBIC_{}{}_{}x{}, vertex_number_{}'.format(vid, frame, height, 2*height,  (img>0).sum()))
    plt.colorbar()
    plt.savefig('BICUBIC_{}{}_{}x{}'.format(vid, frame, height,2*height))
    plt.cla()
    '''
    #image = image.resize((2048,1024), Image.NEAREST)
    #image = image.resize((2048,1024), Image.BILINEAR)
    #image = image.resize((2048,1024), Image.BICUBIC)

    '''
    #gaussian filter!
    r=1 #模版半径，自己自由调整
    s=2 #sigema数值，自己自由调整
    GBlur=MyGaussianBlur(radius=r, sigema=s)#声明高斯模糊类
    temp=GBlur.template()#得到滤波模版
    #im=Image.open('lena1.bmp')#打开图片
    image=GBlur.filter(img, temp)#高斯模糊滤波，得到新的图片
    #image.show()#图片显示

    plt.imshow(image)
    plt.title('Gaussianblur_{}{}_{}x{}, vertex_number_{}'.format(vid, frame, height, 2*height,  (img>0).sum()))
    plt.colorbar()
    plt.savefig('Gaussianblur_{}{}_{}x{}'.format(vid, frame, height,2*height))
    plt.cla()
    pdb.set_trace()
    '''
    return img

def cal_all_map(test_set, height, SAMPLING_TYPE):
    H = 480
    W = 640
    fixmap_all = np.zeros([H, W], dtype=np.float32)
    if SAMPLING_TYPE.split("_")[0] == "GICOPix":
        fixmap_all = np.zeros([10242], dtype=np.float32)
    for vid in test_set:
        fix_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/fixation_largescale", vid)
        for iFrame in range(len(os.listdir(fix_root))):
            fixmap_gt = cv2.imread(os.path.join(fix_root, f'fixmap_f_{iFrame}.png'))
            fixmap_gt = cv2.cvtColor(fixmap_gt, cv2.COLOR_BGR2GRAY)
            if SAMPLING_TYPE.split("_")[0] == "GICOPix":
                fixmap_gt = ErpToSphere(unifS, fixmap_gt)
            fixmap_all += fixmap_gt >0 #yq to do 
    return fixmap_all

def calc_other_map(fixmap_all, fixmap2):
    fixmap_all = (fixmap_all>0).astype(int)
    fixmap2 = (fixmap2>0).astype(int)
    other_map = fixmap_all-fixmap2
    return other_map


def metric_chao(data_set, final_pth, predict_path, act):    
    keys_order = ['AUC_Judd','AUC_Borji', 'NSS', 'CC', 'SIM', 'KLD', "AUC_shuffled"]

    metrics = {
        "AUC_Judd": [AUC_Judd, False, 'fix'],  # Binary fixation map
        "AUC_Borji": [AUC_Borji, False, 'fix'],  # Â Binary fixation map
        "NSS": [NSS, False, 'fix'],  # Â Binary fixation map
        "CC": [CC, False, 'sal'],  # Â Saliency map
        "SIM": [SIM, False, 'sal'],  # Â Saliency map
        "KLD": [KLD, False, 'sal'],
        "AUC_shuffled": [AUC_shuffled, True, 'fix']}  # Â Saliency map

    SAMPLING_TYPE = [  # Â Different sampling method to apply to saliency maps
        "Sphere_9999999",  # Too many points
        "Sphere_1256637",  # 100,000 points per steradian
        "Sphere_10000",  # 10,000
        "GICOPix_10242", #10242
        "Sin",  # Sin(height)
    ]
    SAMPLING_TYPE = SAMPLING_TYPE[-1]  # Â Sin weighting by default
    print("SAMPLING_TYPE: ", SAMPLING_TYPE)
    
    test_set = ['idLVnagjl_s', 'ey9J7w98wlI', 'kZB3KMhqqyI', 'MzcdEI-tSUc', '8ESEI0bqrJ4','1An41lDIJ6Q','6QUCaLvQ_3I', '8feS1rNYEbg','ByBF08H-wDA','fryDy9YcbI4', 'RbgxpagCY_c_2','dd39herpgXA']

    height = 480
    width = 640
    house, _ , _, _, _ = load_graph_info(5)
    point_theta = []
    for i in range(len(house.vertices)):
        point_theta.append(house.vertices[i].rtf)

    unifS, VerticalWeighting = unifs_generate(SAMPLING_TYPE, point_theta, height)
    fixmap_all = cal_all_map(test_set, height, SAMPLING_TYPE)
    save_pth_name = final_pth.split('.')[0]
    pre_graph_path = os.path.join(predict_path, f'{data_set}_{save_pth_name}_{act}_presal.npy')
    
    with open(f"{predict_path}/metric.csv", "a") as saveFile:
        saveFile.write("Vid, NSS, CC, KLD, AUC_B, AUC_S, AUC_J, SIM, All\n")
        
        #predicted saliency
        salmap_pr = np.load(pre_graph_path)
        KL = 0
        AUC_S = 0
        AUC_B = 0
        AUC_J = 0
        CCC = 0
        NSSS = 0
        SIMM = 0
        salmap1_idx = 0

        for vid in test_set:
            print(vid)
            #label
            sal_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/saliency", vid) 
            fix_root = os.path.join("/home/yq/Audio/Audio_visual/data/Chao/fixation_largescale", vid)

            KL_vid = 0
            AUC_S_vid = 0
            AUC_B_vid = 0
            AUC_J_vid = 0
            CCC_vid = 0
            NSSS_vid = 0
            SIMM_vid = 0
            salmap_idx_vid = 0
            for iFrame in range(len(os.listdir(sal_root))): 
                    salmap_idx_vid += 1
                    #predicted saliency map
                    print(salmap_pr.shape, salmap1_idx)
                    salmap_pred = salmap_pr[salmap1_idx] #10242
                    salmap1_idx = salmap1_idx+1

                    
                    #label
                    salmap_gt = cv2.imread(os.path.join(sal_root, f'salmap_f_{iFrame}.png'))
                    salmap_gt = cv2.cvtColor(salmap_gt, cv2.COLOR_BGR2GRAY)
                    salmap_gt = cv2.resize(salmap_gt, (640, 480))

                    fixmap_gt = cv2.imread(os.path.join(fix_root, f'fixmap_f_{iFrame}.png'))
                    fixmap_gt = cv2.cvtColor(fixmap_gt, cv2.COLOR_BGR2GRAY)


                    # Weight saliency maps vertically if specified
                    if SAMPLING_TYPE == "Sin":
                        salmap_pred = SphereToErp(height, salmap_pred, point_theta)
                        
                        salmap_pred = salmap_pred * VerticalWeighting[:, None] + EPSILON
                        salmap_gt = salmap_gt * VerticalWeighting[:, None] + EPSILON

                    elif SAMPLING_TYPE.split("_")[0] == "GICOPix":
                        salmap_gt = ErpToSphere(unifS, salmap_gt)
                        fixmap_gt = ErpToSphere(unifS, fixmap_gt)

                    # Apply uniform sphere sampling if specified
                    elif SAMPLING_TYPE.split("_")[0] == "Sphere":
                        salmap_pred = SphereToErp(height, salmap_pred, point_theta)
                        salmap_pred = salmap_pred[unifS[:, 0], unifS[:, 1]]
                        salmap_gt = salmap_gt[unifS[:, 0], unifS[:, 1]]
                        fixmap_gt = fixmap_gt[unifS[:, 0], unifS[:, 1]]

                    ###predicted
                    salmap_pred = normalize(salmap_pred, method='sum')

                    ###label
                    salmap_gt = normalize(salmap_gt, method='sum')
                    other_map = calc_other_map(fixmap_all, fixmap_gt) #all the other maps of the dataset

                    values = getSimVal(keys_order,metrics,salmap_pred, salmap_gt, fixmap2=fixmap_gt, other_map2=other_map)
                    
                    KL_vid += values[5]
                    AUC_S_vid += values[6]
                    AUC_B_vid += values[1]
                    AUC_J_vid += values[0]
                    NSSS_vid += values[2]
                    CCC_vid += values[3]
                    SIMM_vid += values[4]

                    KL += values[5]
                    AUC_S += values[6]
                    AUC_B += values[1]
                    AUC_J += values[0]
                    NSSS += values[2]
                    CCC += values[3]
                    SIMM += values[4]
            saveFile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(vid, round(NSSS_vid/salmap_idx_vid, 3), round(CCC_vid/salmap_idx_vid, 3), round(KL_vid/salmap_idx_vid, 3), round(AUC_B_vid/salmap_idx_vid, 3), round(AUC_S_vid/salmap_idx_vid, 3), round(AUC_J_vid/salmap_idx_vid,3), round(SIMM_vid/salmap_idx_vid, 3), round(NSSS_vid/salmap_idx_vid+5*CCC_vid/salmap_idx_vid-2*KL_vid/salmap_idx_vid, 3)))

        # Outputs results
        print("stimName, metric, value")
        print('KL:',round(KL/salmap1_idx, 3))
        print('AUC_S:', round(AUC_S/salmap1_idx, 3))
        print('AUC_J:', round(AUC_J/salmap1_idx, 3))
        print('AUC_B:', round(AUC_B/salmap1_idx,3))
        print('SIM:', round(SIMM/salmap1_idx,3))
        print('CC:', round(CCC/salmap1_idx,3))
        print('NSS:', round(NSSS/salmap1_idx,3))
        saveFile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format('overall', round(NSSS/salmap1_idx, 3), round(CCC/salmap1_idx, 3), round(KL/salmap1_idx, 3), round(AUC_B/salmap1_idx, 3), round(AUC_S/salmap1_idx, 3), round(AUC_J/salmap1_idx,3), round(SIMM/salmap1_idx, 3), round(NSSS/salmap1_idx+5*CCC/salmap1_idx-2*KL/salmap1_idx, 3)))
    return round(KL/salmap1_idx, 3), round(NSSS/salmap1_idx, 3), round(CCC/salmap1_idx,3), round(NSSS/salmap1_idx+5*CCC/salmap1_idx-2*KL/salmap1_idx, 3)
