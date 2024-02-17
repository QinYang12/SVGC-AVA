import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import scipy.io
import numpy as np
from graph_basic import S_Graph
import copy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gl', '--graph_level', default=5, type=int)
parser.add_argument('-edges', '--edge_number', default=6, type=int)
parser.add_argument('-metric', '--metric', default='euclidean', type=str)
args = parser.parse_args()

def grid_sphere(graph_level):
    house = S_Graph(graph_level, 'fa')
    z = []
    z_theta = []
    for j in house.vertices:
            theta = house.vertices[j].rtf[0] #-0.5~0.5
            phi = house.vertices[j].rtf[1] #-1~1
            z.append(house.vertices[j].xyz)
            z_theta.append([theta, phi])
    return z,z_theta

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)#这时候z的形状是（点数，3）
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]#对d的每一行进行从小到大的排序并且返回距离最小的k个点的位置，这里实际上就是表示的两点之间的连接关系
    idxfac=copy.copy(idx)
    m = np.sqrt(50 - 10 * np.sqrt(5)) / 10
    n = np.sqrt(50 + 10 * np.sqrt(5)) / 10
    v_base = [[-m, 0, n], [m, 0, n], [-m, 0, -n], [m, 0, -n], [0, n, m], [0, n, -m], [0, -n, m], [0, -n, -m], [n, m, 0],
              [-n, m, 0], [n, -m, 0], [-n, -m, 0]]

    temp = []
    z=z.tolist()
    for i in range(12):
        temp.append(z.index(v_base[i]))#这里存着z中原始12个节点的索引
    for i in temp:
        idxfac[i][-1] = -1
        #for j in  range(idx.shape[1]):
    z=np.array(z)
    d.sort()#对d的每一行进行从小到大的排序

    d = d[:, 1:k+1]
    # aa=0
    # bb=0
    # print("d", d)
    # for i in range(d.shape[0]):
    #
    #     if d[i][0]/(d[i][-1] - d[i][4]) >= 1:
    #         aa = aa + 1
    #     if d[i][-1] - d[i][4] < 0.01:
    #         bb = bb + 1
    # print('a+b=',aa+bb)
    # print('a',aa)
    # print('b',bb)
    return d, idx,idxfac


def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0

    # Weights.
    sigma2 = np.mean(dist[:, -1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def laplacian(W, normalized=True):
    """Return the Laplacian of the weigth matrix."""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = scipy.sparse.diags(d.A.squeeze(), 0)
        I = scipy.sparse.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is scipy.sparse.csr.csr_matrix
    return L

def lmax(L, normalized=True):
    """Upper-bound on the spectrum."""
    if normalized:
        return 2
    else:
        return scipy.sparse.linalg.eigsh(
                L, k=1, which='LM', return_eigenvectors=False)[0]

def grid_graph(level, corners=False):
    z, z_theta = grid_sphere(level)  
    print('z shape is' + str(np.array(z).shape) + 'z_theta shape is' + str(np.array(z_theta).shape))
    dist, idx, idxfac = distance_sklearn_metrics(np.array(z), k=args.edge_number, metric=args.metric)
    A = adjacency(dist, idx)
    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max() / 1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz // 2, args.edge_number * level ** 2 // 2))
    return z, z_theta, A, idxfac


def graph_construct():
    graphs_adjacency = []
    for level in range(args.graph_level, 0, -1):  # 点数从多到少
        z, z_theta, A, idxfac = grid_graph(level, corners=False) 
        graphs_adjacency.append(A)
    L = [laplacian(A, normalized=True) for A in graphs_adjacency]
    print(len(L))
    scipy.io.savemat("graph_laplacian.mat", {'graph_laplacian':L})

if __name__=="__main__":
    graph_construct()

