import numpy as np
from PIL import Image
import pdb, sys, os, time

v_num_dict = []
for i in range(12):
    v_num_dict.append(10*2**(2*i)+2)
# v_num_dict = [12,42,162,642,2562,10242,40962,163842,655362,2621442,10485762,41943042]
def timer(func):
    def warper(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        dur = time.time() - tic
        print("call {} used {:0.2f}s".format(func.__name__, dur))
        return res
    return warper


class Vertice:
    vCount = 0
    def __init__(self, index):
        self.index = index
        self.rgb = None
        self.gray = 0
        self.sal = 0
        self.xyz = None
        self.rtf = None
        self.hashxyz = None
        self.neighboridxs = []
        self.neighboragls = []
        self.affiliation = []
        self.max_depth = 3
        self.wp_xyz = []
        self.neip_xyz = []
        Vertice.vCount += 1

    def init_v(self, f1, f2):
        bits = 100000
        # f1 f2是两个father节点，这个函数的功能是从两个father节点生出一个son节点
        # 原来，f1和f2是邻居，要把这两个的邻居对应换成新加节点
        # 新加节点的第0和第1个邻居分别换成f1 f2
        # 根据f1 f2的xyz坐标算出新加节点的xyz和rtf坐标

        # 操作邻居
        self.neighboridxs.append(f1.index)  # 新加节点邻居加上f1
        self.neighboridxs.append(f2.index)  # 新加节点邻居加上f2
        tmp = f1.neighboridxs.index(f2.index)  # f2在f1的neighbors里面的排号
        f1.neighboridxs[tmp] = self.index  # f1的邻居由f2替换成了新加节点
        tmp = f2.neighboridxs.index(f1.index)  # f1在f2的neighbors里面的排号
        f2.neighboridxs[tmp] = self.index  # f2的邻居由f1换成新节点

        # 算这个新加节点的xyz归一化坐标
        self.xyz = get_sum(f1.xyz, f2.xyz)
        r = get_length(self.xyz)
        self.xyz = lin_product(self.xyz, 1/r)
        x, y, z = self.xyz

        # 算这个新加节点的theta phi
        # 这里也可以用arccos来算phi, 但是需要知道小圆的半径，可以在上一步算r的时候用np.hypot算。
        theta = 0.5 - np.arccos(z)/np.pi  # theta最高处是0.5，最低处是-0.5
        phi = np.mod(np.arctan2(y,x)/np.pi,2) - 1  # phi 从(1, 0)开始是-1，逆时针转一圈成1
        self.rtf = [theta, phi]

        # 算这个新加节点的哈希值，设计了这个函数，得到结果是hashable的
        self.hashxyz = int(x*bits)*bits*bits + int(y*bits)*bits + int(z*bits)

        # 新加节点的精度
        if -0.2 < theta < 0.2 and -0.3 < phi < 0.3:
            self.max_depth = 12

        # 新加节点的 affiliation
        # 这个相当于是每个点的索引，知道每个点是一层一层根据谁生出来的
        # 2021.1.11添加的这个功能。之前用到类似功能，可能还是根据哈希值一点点改的，暂时不处理
        if f1.affiliation == f2.affiliation == []:
            self.affiliation.append(sorted([f1.index, f2.index]))
        elif f1.affiliation == []:
            self.affiliation = f2.affiliation.copy()
            self.affiliation.append([f1.index, f2.index])
        elif f2.affiliation == []:
            self.affiliation = f1.affiliation.copy()
            self.affiliation.append([f2.index, f1.index])
        else:
            len1, len2 = sorted([len(f1.affiliation), len(f2.affiliation)])
            for dpt in range(len1):
                tmp = calculate_affiliation(f1.affiliation[dpt], f2.affiliation[dpt])
                self.affiliation.append(tmp)
            if len(f1.affiliation) < len(f2.affiliation):
                self.affiliation = self.affiliation + f2.affiliation[len1:len2]
                self.affiliation.append([f1.index, f2.index])
            elif len(f2.affiliation) < len(f1.affiliation):
                self.affiliation = self.affiliation + f1.affiliation[len1:len2]
                self.affiliation.append([f2.index, f1.index])
            else:
                self.affiliation.append(sorted([f1.index, f2.index]))

    def get_relative_angle(self, subject):
        # 这个函数的功能是算self -> subject 的相对角度，头顶是0，从球里看逆时针转一圈变成2，球外顺时针
        # subject 直接是xyz坐标
        x, y, z = self.xyz
        r_xy = np.hypot(x, y)
        if r_xy == 0:
            if z == -1.0:
                return np.mod(np.arctan2(subject[1],subject[0])/np.pi,2)
            elif z == 1.0:
                return np.mod(np.arctan2(subject[1],-subject[0])/np.pi,2)
        x0 = x/r_xy
        y0 = y/r_xy
        sub_plane_x0 = [-x0*z, -y0*z, r_xy]
        sub_plane_y0 = [-y0, x0, 0]
        # print(sub_plane_x0, sub_plane_y0)
        spx = dot_product(sub_plane_x0, subject)
        spy = dot_product(sub_plane_y0, subject)
        #print(spx, spy)
        return np.mod(np.arctan2(spy,spx)/np.pi,2)  # 从（1，0）(1,0是正上方)开始是0，从球里看逆时针转一圈变成2，球外顺时针

    def sort_neighbors(self, real_neighbors):
        # 这个函数就是对一个点的所有邻居排序
        # 这里的real_neighbors是一个list，元素是Vertice。不同于Vertice.neighbours，元素是index
        tem = []
        for neighbor in real_neighbors:
            tem.append((neighbor.index, self.get_relative_angle(neighbor.xyz)))
        tem = sorted(tem,key=lambda x: x[1]) #按照第二列，也就是相对角度排序
        sorted_neighbors = []
        neighbor_angles = []
        for t in tem:
            sorted_neighbors.append(t[0])
            neighbor_angles.append(t[1])
        #print(i, sorted_neighbors)
        self.neighboridxs = sorted_neighbors
        self.neighboragls = neighbor_angles

    def get_average_distance(self, house, neighboridxs):
        #x, y, z = self.xyz
        distance = 0
        n = 0
        neip_xyz = []
        for neighbor in neighboridxs:
            #x_n,y_n,z_n = neighbor.xyz
            neip_xyz.append((np.array(house[neighbor].xyz)-np.array(self.xyz)).tolist())
            dis = get_distance(self.xyz, house[neighbor].xyz)
            distance += dis
            n += 1
        self.neip_xyz = neip_xyz
        distance = distance/n 

        return distance
        #self.neip_dis = (neip_dis/distance).tolist()  #d_average/d_x
    
    def get_w_position(self, distance):        
        
        angle_deta = 2*np.arcsin(distance/2)/np.pi #r=1, angle_deta:-pi~pi
        theta, phi = self.rtf
        
        #four kernel
        theta1 = theta+angle_deta
        phi1 = phi
        theta2 = theta-angle_deta
        phi2 = phi
        if theta+angle_deta>0.5:
            theta1 = 1-(theta+angle_deta)
            phi1 = np.mod(phi, 2)-1
        if theta-angle_deta<-0.5:
            theta2 = -1-(theta-angle_deta)
            phi2 = np.mod(phi, 2)-1
        wp_rtf = np.array([[theta1, phi1], [theta, np.mod(phi+angle_deta+1,2)-1], [theta2, phi2], [theta, np.mod(phi-angle_deta+1, 2)-1]])
        
        #rtf--xyz
        theta = 0.5-wp_rtf[:, 0] #0.5~-0.5  >> 0~1
        phi = wp_rtf[:, 1]+1 #-1~1 >> 0~2
        x = sin(theta)*cos(phi) #r=1
        y = sin(theta)*sin(phi)
        z = cos(theta)
        wp_xyz = np.concatenate((x[:, np.newaxis],y[:, np.newaxis],z[:, np.newaxis]), axis=1) # 4*3
        self.wp_xyz = (wp_xyz-self.xyz).tolist() 



class S_Graph:
    def __init__(self, depth, method='fa'):
        self.depth = depth
        self.v_num = v_num_dict[depth]
        self.vertices = {}
        self.hashxyzmap = {}
        self.f_init = [[1,4,0],[4,9,0],[4,5,9],[8,5,4],[1,8,4],
                      [1,10,8],[10,3,8],[8,3,5],[3,2,5],[3,7,2],
                      [3,10,7],[10,6,7],[6,11,7],[6,0,11],[6,1,0],
                      [10,1,6],[11,0,9],[2,11,9],[5,2,9],[11,2,7]]

        print("Generating graph")
        print('depth =', self.depth, ' v_Number =', self.v_num)
        self.init_g(method)

    @timer
    def init_g(self, method):
        bits = 100000
        m = np.sqrt(50-10*np.sqrt(5))/10
        n = np.sqrt(50+10*np.sqrt(5))/10
        # 初始12个点的xyz坐标
        v_init = [[-m,0,n],[m,0,n],[-m,0,-n],[m,0,-n],[0,n,m],[0,n,-m],
                    [0,-n,m],[0,-n,-m],[n,m,0],[-n,m,0],[n,-m,0],[-n,-m,0]]
        # 初始30条边（用12*5的邻接列表来表示）
        e_init = [[1,4,6,9,11],[0,4,6,8,10],[3,5,7,9,11],[2,5,7,8,10],
                  [0,1,5,8,9],[2,3,4,8,9],[0,1,7,10,11],[2,3,6,10,11],
                    [1,3,4,5,10],[0,2,4,5,11],[1,3,6,7,8],[0,2,6,7,9]]

        for i in range(12):
            self.vertices[i] = Vertice(i)
            self.vertices[i].xyz = v_init[i]
            self.vertices[i].neighboridxs = e_init[i]
            # 算这个新加节点的theta phi
            x, y, z = v_init[i]
            theta = 0.5 - np.arccos(z)/np.pi  # theta最高处是0.5，最低处是-0.5
            phi = np.mod(np.arctan2(y,x)/np.pi,2) - 1  # phi从(1, 0)开始是-1，转一圈成1
            self.vertices[i].rtf = [theta, phi]
            self.vertices[i].max_depth = 12
            self.vertices[i].hashxyz = int(x*bits)*bits*bits + int(y*bits)*bits + int(z*bits)
            self.hashxyzmap[self.vertices[i].hashxyz] = i

        if method == 'fa':
            self.fully_subdivide_angular()

    def fully_subdivide_angular(self):
        # 这个是用到了角度信息进行球面的fully subdivide
        # 先排序最开始的12个
        for i in range (12):
            real_neighbors = []
            for neighbor in self.vertices[i].neighboridxs:  # 这里的neighbor是index
                real_neighbors.append(self.vertices[neighbor])
            self.vertices[i].sort_neighbors(real_neighbors)

        for depth in range(self.depth):  # 一层一层来，每层分三步走。第一步插入，第二步链接，第三部对新一层节点排序
            print('current layer:', depth+1, end='   \r')

            # 第一步是把每个edge打断，中点插入新的点，并变成两个新的edge
            index = v_num_dict[depth]  # 新加层的起始index（刚好等于上一层的最大节点个数+1）
            for i in range(v_num_dict[depth]):
            # 对father层每个节点的遍历，i等于当前遍历到的节点的index
                for neighbor in self.vertices[i].neighboridxs:
                # 对当前节点的邻居遍历
                    if neighbor < v_num_dict[depth]:  # 新加的点不需要打断，只打断原来的
                        self.vertices[index] = Vertice(index)  # 创建这个新加节点
                        self.vertices[index].init_v(self.vertices[i], self.vertices[neighbor])
                        self.hashxyzmap[self.vertices[index].hashxyz] = index
                        index += 1  # 下一个新加节点index加一

            # 第二步是每个father层的节点的所有新邻居直接按顺序连成一圈
            for i in range(v_num_dict[depth]):
                points_to_add_edge = self.vertices[i].neighboridxs
                length = len(points_to_add_edge)
                for j in range(length-1):
                    self.vertices[points_to_add_edge[j]].neighboridxs.append(points_to_add_edge[j+1])
                    self.vertices[points_to_add_edge[j+1]].neighboridxs.append(points_to_add_edge[j])
                self.vertices[points_to_add_edge[0]].neighboridxs.append(points_to_add_edge[length-1])
                self.vertices[points_to_add_edge[length-1]].neighboridxs.append(points_to_add_edge[0])

            # 第三步是对这一轮生成的子节点排序
            for i in range(v_num_dict[depth], v_num_dict[depth+1]):
                real_neighbors = []
                for neighbor in self.vertices[i].neighboridxs:  # 这里的neighbor是index
                    real_neighbors.append(self.vertices[neighbor])
                self.vertices[i].sort_neighbors(real_neighbors)

        #构图完成后 计算所有点wp, neighborP(neip)
        for i in range(v_num_dict[self.depth]):
            #get the neighbor distance
            average_distance = self.vertices[i].get_average_distance(self.vertices, self.vertices[i].neighboridxs) 
            #get w_position
            self.vertices[i].get_w_position(average_distance)



    def Node_correspondence(self, imgs):#返回图中的每个点的像素值
        if len(imgs.shape)==3:
            H, W, img_channel = imgs.shape
        elif len(imgs.shape)<3:
            H, W = imgs.shape
        pixels = []
        for j in self.vertices:
            theta = self.vertices[j].rtf[0] #-0.5~0.5
            phi = self.vertices[j].rtf[1] #-1~1
            x=(W-1)*0.5*(phi+1)
            y=(H-1)*(0.5-theta)
            x_int=int(x)      #int()是向下取整
            y_int=int(y)
            dx = x-x_int
            dy = y-y_int
            pixel_bo_left = imgs[min(y_int+1,H-1), x_int]
            pixel_bo_right = imgs[min(y_int+1,H-1),min(x_int+1,W-1)]
            pixel_up_left = imgs[y_int,x_int]
            pixel_up_right = imgs[y_int,min(x_int+1,W-1)]
            pixel_bo = (1-dx) * pixel_bo_left + (dx) * pixel_bo_right
            pixel_up = (1-dx) * pixel_up_left + (dx) * pixel_up_right
            pixel = (1-dy) * pixel_up + (dy) * pixel_bo
            pixels.append(pixel)
        if (i+1)%10==0:
            print('project_data for {}/{} done.'.format(i+1,img_num))
        return np.array(pixels)

    def from_img(self, img=None, lookup_angle=0.2, rotate='nr', mode='RGB', debug=False):
        # load image and convert format to h*w*3 or h*w
        assert rotate in ['nr', 'tr', 'sr', 'fr', 'r'] or '_' in rotate
        # no / tiny / small(SGCN) / fit / rotate
        # rotate = "45_90" ---> (0.25, 0.5)
        if '_' in rotate:
            theta_phi = rotate.split('_')
            theta_ctr = float(theta_phi[0]) / 180
            phi_ctr = float(theta_phi[1]) / 180
            assert -0.5 <= theta_ctr <= 0.5
            assert -1 <= phi_ctr <= 1
        assert mode in ['RGB', 'L']
        if type(img) == str:
            img_name = img
            print(img_name, end=', ')
            img = Image.open(img_name)
        img = np.array(img).astype(float).squeeze()
        assert img.ndim <= 3
        if mode == 'RGB':
            if img.ndim == 3:
                if img.shape[0] in [3, 4]:
                    img = img.transpose(1,2,0)
                img = img[:, :, :3]
            if img.ndim == 2:
                img = np.expand_dims(img,2).repeat(3, axis=2)
        if mode == "L":
            if img.ndim == 3:
                if img.shape[0] in [3, 4]:
                    img = img.transpose(1,2,0)
                img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114

        # load positional information
        # theta 是纬度, phi 是经度
        img_h, img_w = img.shape[0], img.shape[1]
        if rotate == 'nr':
            theta_ctr, phi_ctr = 0, 0
        elif rotate == 'tr':
            theta_ctr = np.random.rand()*0.1 - 0.05
            phi_ctr = np.random.rand()*0.1 - 0.05
        elif rotate == 'sr':
            theta_ctr = np.random.rand()*0.5 - 0.25
            phi_ctr = np.random.rand() - 0.5
        elif rotate == 'fr':
            theta_max = 0.5 - lookup_angle
            theta_ctr = np.random.rand()*2*theta_max - theta_max
            phi_ctr = np.random.rand()*2 - 1
        elif rotate == 'r':
            theta_ctr = np.random.rand() - 0.5
            phi_ctr = np.random.rand()*2 - 1
        radius = img_h / 2 / tan(lookup_angle)

        # start to project
        theta = np.zeros(len(self.vertices))
        phi = np.zeros(len(self.vertices))
        rgbs = np.zeros((len(self.vertices), 3))
        grays = np.zeros(len(self.vertices))

        for i in self.vertices:
            theta[i], phi[i] = self.vertices[i].rtf[0], self.vertices[i].rtf[1]

        agl = sin(theta_ctr)*sin(theta)+cos(theta_ctr)*cos(theta)*cos(phi-phi_ctr)
        x = radius*cos(theta)*sin(phi-phi_ctr)/agl
        y = radius*(cos(theta_ctr)*sin(theta)-sin(theta_ctr)*cos(theta)*cos(phi-phi_ctr))/agl
        prj_points = np.where((abs(x)<=img_w/2) & (abs(y)<=img_w/2) & (agl>0))

        x, y = x[prj_points], y[prj_points]
        x, y = x+(img_w-1)/2, (img_h-1)/2-y
        x, y = np.where(x>0, x, 0), np.where(y>0, y, 0)
        x_int, y_int = x.astype(np.int32), y.astype(np.int32)
        dx, dy = x-x_int, y-y_int
        if mode == "RGB":
            dx, dy = np.expand_dims(dx,1), np.expand_dims(dy,1)
        xp, yp = x_int+1, y_int+1
        xp, yp = np.where(xp<img_w-1, xp, img_w-1), np.where(yp<img_h-1, yp, img_h-1)

        pixel_up_left  = img[y_int, x_int]
        pixel_up_right = img[y_int, xp]
        pixel_bo_left  = img[yp, x_int]
        pixel_bo_right = img[yp, xp]
        pixel_bo = (1-dx) * pixel_bo_left + dx * pixel_bo_right
        pixel_up = (1-dx) * pixel_up_left + dx * pixel_up_right
        pixel = (1-dy) * pixel_up + dy * pixel_bo

        if debug:
            print(f"projected {len(pixel)} / {self.v_num} points")
            # pdb.set_trace()

        if mode == 'RGB':
            rgbs[prj_points] = pixel
            grays = rgbs[:,0]*0.299 + rgbs[:,1]*0.587 + rgbs[:,2]*0.114
            for i in self.vertices:
                self.vertices[i].rgb, self.vertices[i].gray = rgbs[i], grays[i]
            return rgbs
        elif mode == 'L':
            grays[prj_points] = pixel
            rgbs = np.expand_dims(grays,1).repeat(3, axis=1)
            for i in self.vertices:
                self.vertices[i].rgb, self.vertices[i].gray = rgbs[i], grays[i]
            return grays

    def show_vertices(self, color=None, cmap='gray'):
        import pyvista as pv
        # 如果指定color 则是全场一个颜色
        # 如果color = None 那就看cmap。如果cmap是bwr的话，就直接显示红蓝图，不用想了
        # 如果cmap是gray，那要看一下图像是否给投影了gray数据，如果投了就显示灰度图，如果没投就要42个黑点和剩下白点
        v_xyz = []
        v_gray = []
        v_sal = []
        for i in self.vertices:
            v_xyz.append(self.vertices[i].xyz)
            v_gray.append(self.vertices[i].gray)
            v_sal.append(self.vertices[i].sal)
            #print(self.vertices[i].neighboridxs)
        print("Actual number of vertices:", len(v_xyz))
        v_xyz = np.array(v_xyz)
        v_gray = np.array(v_gray)
        v_sal = np.array(v_sal)
        v_arange = np.arange(len(v_sal))
        #if v_gray.max() == 0:
        #    v_gray[42:] = 255
        sphere = pv.PolyData(v_xyz)
        if cmap == 'gray':
            sphere.point_arrays['scalars'] = v_gray
        elif cmap == 'bwr':
            sphere.point_arrays['scalars'] = v_arange
        sphere.set_active_scalars('scalars')
        p = pv.Plotter()
        p.add_mesh(sphere, color=color, cmap=cmap, point_size=15, render_points_as_spheres=True)

        point = pv.PolyData(np.array([[0,0,1], [0,1,0], [1,0,0]]))
        #p.add_mesh(point, color='r', point_size=15, render_points_as_spheres=True)
        p.add_axes()
        p.show()


def cos(x):
    return np.cos(x*np.pi)
def sin(x):
    return np.sin(x*np.pi)
def tan(x):
    return np.tan(x*np.pi)
def get_distance(v1, v2):
    return np.sqrt((v1[0]-v2[0])**2+(v1[1]-v2[1])**2+(v1[2]-v2[2])**2)
def get_length(v):
    return get_distance(v, [0,0,0])
def get_sum(v1, v2):
    return [v2[0]+v1[0], v2[1]+v1[1], v2[2]+v1[2]]
def dot_product(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
def lin_product(v, l):
    return [v[0]*l, v[1]*l, v[2]*l]
def calculate_affiliation(array1, array2):
    if len(array1) == 3:
        return array1
    elif len(array2) == 3:
        return array2
    elif array1 == array2:
        return array1
    else:
        return sorted(list(set(array1 + array2)))

if __name__ == '__main__':
    pdb.set_trace()
    import torchvision
    from torchvision import transforms
    act = 'train'
    dataset = torchvision.datasets.MNIST('data', act=='train', download=True,
              transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    house = S_Graph(5)
    house.from_img(dataset[0][0], debug=True, mode="L")
    house.show_vertices()
