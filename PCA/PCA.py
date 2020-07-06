import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def load_data(datafile):
    file = open(datafile, 'r')
    X=[]
    for line in file.readlines():
        line = line.rstrip('\n')
        line = line.split(' ')
        for i in range(len(line)):
            line[i] = float(line[i])
        X.append(line)
    file.close()
    X=np.array(X)
    return X

def load_tensor(filepath,pointnum):
    X=[]
    for i in range(pointnum): # 10000
        data = np.load(filepath+'/arr_{}.npz'.format(i))
        x = data['arr_0']
        # x=x.ravel()
        data.close()
        X.append(x)
    X=np.array(X)
    return X


def get_X(data,pointnum,layernum=7):
    if data=='naive':
        X = np.array(
            [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
    elif data=='txt':
        datafile = 'data.txt'
        X = load_data(datafile)
    elif data=='npz':
        filepath='../1_project/LeNet/y1'
        X=load_tensor(filepath,pointnum)
    return X


def pca_process(X,n_components=2):
    pca = PCA(n_components=n_components)   #降到2维
    pca.fit(X)                  #训练
    newX=pca.fit_transform(X)   #降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_ratio_)  #输出贡献率
    print(newX)                  #输出降维后的数据
    return newX

def trans_y2color(y,num=10):
    if num==10:
        color_dict={1:'red',2:'orange',3:'yellow',4:'lawngreen',5:'darkgreen',
                    6:'aquamarine',7:'deepskyblue',8:'navy',9:'darkorchid',0:'fuchsia'}
    else:
        raise Exception('class number wrong when transfer color')
    y_color=[]
    for i in range(len(y)):
        y_color.append(color_dict[y[i]])
    return y_color


def clustering(cluster,newX,cluster_number=10):
    if cluster=='GMM':
        ##设置gmm函数
        gmm = GaussianMixture(n_components=cluster_number, covariance_type='full').fit(newX)
        ##训练数据
        y_pred = gmm.predict(newX)
        y_pred=trans_y2color(y_pred,cluster_number)
        ax = plt.figure().add_subplot(111, projection='3d')
        area = np.pi * 4 ** 2
        ax.scatter(newX[:, 0], newX[:, 1], newX[:, 2], c=y_pred, s=area, alpha=0.4)

        # plt.scatter(newX[:, 0], newX[:, 1], c=y_pred)
        # plt.savefig('layer_{}_GMM'.format(layernum))
        plt.show()
    elif cluster=='KMeans':
        # 开始调用函数聚类
        cls = KMeans(cluster_number).fit(newX)

        # 输出X中每项所属分类的一个列表
        y=cls.labels_
        y=trans_y2color(y,cluster_number)
        ax = plt.figure().add_subplot(111, projection='3d')
        area = np.pi * 4 ** 2
        ax.scatter(newX[:,0], newX[:,1], newX[:,2], c=y, s=area, alpha=0.4)
        # plt.scatter(newX[:, 0], newX[:, 1], c=y)
        # plt.savefig('layer_{}_KMeans'.format(layernum))
        plt.show()

data='npz' # 'naive' or 'txt' or 'npz'
cluster='KMeans' # 'GMM' or 'KMeans'
cluster_number=10
pointnum=1000
layernum=7
X=get_X(data,pointnum,layernum)

newX=pca_process(X,n_components=3)
#clustering(cluster,newX)

def get_label(filepath):
    file=open(filepath,'r')
    y=[]
    for line in file.readlines():
        line=line.rstrip('\n')
        line=line.split('\t')
        y.append(int(line[-1]))
    file.close()
    return y

y=get_label('../1_project/LeNet/label.txt')
y=y[:pointnum]
y=trans_y2color(y,10)
ax = plt.figure().add_subplot(111, projection='3d')
area = np.pi * 4 ** 2
ax.scatter(newX[:,0], newX[:,1], newX[:,2], c=y, s=area, alpha=0.4)

# plt.scatter(newX[:, 0], newX[:, 1], c=y)
# plt.savefig('layer_{}_label'.format(layernum))
plt.show()