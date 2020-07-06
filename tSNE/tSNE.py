import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

def get_label(filepath):
    file=open(filepath,'r')
    y=[]
    for line in file.readlines():
        line=line.rstrip('\n')
        line=line.split('\t')
        y.append(int(line[-1]))
    file.close()
    return y

def get_X(data,pointnum=100,layernum=7):
    if data=='naive':
        X = np.array(
            [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
    elif data=='txt':
        datafile = '../PCA/data.txt'
        X = load_data(datafile)
    elif data=='npz':
        filepath='../1_project/LeNet/y1'
        X=load_tensor(filepath,pointnum)
    else:
        raise Exception('Illegal setting of data!')
    return X

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


data='npz' # 'naive' or 'txt' or 'npz'
n_components=3
pointnum=1000
classnum=10
layernum=40
X=get_X(data,pointnum,layernum)
tsne = TSNE(n_components=n_components,init='pca', random_state=0)
newX = tsne.fit_transform(X)
y=get_label('../1_project/LeNet/label.txt'.format(layernum))
y=y[:pointnum]
y=trans_y2color(y,classnum)

x_min, x_max = newX.min(0), newX.max(0)
X_norm = (newX - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()
# plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y)
# #plt.savefig('layer_{}_tSNE_init'.format(layernum))

ax = plt.figure().add_subplot(111, projection='3d')
area = np.pi * 4 ** 2
ax.scatter(newX[:,0], newX[:,1], newX[:,2], c=y, s=area, alpha=0.4)
plt.show()