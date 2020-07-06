import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
'''
输入：数据集D = {(x1, y1), (x2, y2), ..., ((xm, ym))}, 其中任意样本xi为n维向量，yi∈{C1, C2, ..., Ck}，降维到的维度d。

输出：降维后的样本集D′

1) 计算类内散度矩阵Sw
2) 计算类间散度矩阵Sb
3) 计算矩阵Sw ^−1 * Sb
4）计算Sw ^−1 * Sb的最大的d个特征值和对应的d个特征向量(w1, w2, ...wd), 得到投影矩阵W
5) 对样本集中的每一个样本特征xi, 转化为新的样本zi = WT * xi
6) 得到输出样本集
'''
def load_data(batch_size=100):
    train_dataset = dsets.MNIST(root = '../../dataset', #选择数据的根目录
                               train = True, # 选择训练集
                               transform = transforms.ToTensor(), #转换成tensor变量
                               download = False) # 不从网络上download图片
    test_dataset = dsets.MNIST(root = '../../dataset', #选择数据的根目录
                               train = False, # 选择训练集
                               transform = transforms.ToTensor(), #转换成tensor变量
                               download = False) # 不从网络上download图片
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size = train_batch
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=test_batch
    )
    return train_loader,test_loader


def data_preprocess(train_loader,test_loader):
    pca = PCA(n_components=feature_num)
    for batch_index, (x, y) in enumerate(train_loader):
        train_x = x.reshape(train_batch, 784)  # 6000X1X28X28 -> 6000X784
        pca.fit(train_x)
        train_x=pca.fit_transform(train_x)
        train_y=y
        break
    for batch_index, (x, y) in enumerate(test_loader):
        test_x = x.reshape(test_batch, 784)  # 1000X1X28X28 -> 1000X784
        test_x=pca.fit_transform(test_x)
        test_y=y
        break
    x_classified=[[],[],[],[],[],[],[],[],[],[]]
    num=[0,0,0,0,0,0,0,0,0,0]
    for i in range(train_batch):
        class_num=train_y[i]
        if num[class_num]==0:
            x_classified[class_num]=train_x[i].reshape(1,feature_num)
        else:
            x_classified[class_num]=np.concatenate((x_classified[class_num],train_x[i].reshape(1,feature_num)),axis=0)
        num[train_y[i]]+=1
    return x_classified, num, train_x, train_y, test_x, test_y

def meanX(data):
    return np.mean(data, axis=0)

def compute_Si(x): # x: ~600X784
    n=np.shape(x)[0]
    mean = meanX(x).reshape(feature_num,1)  # 784 -> 784X1
    Si=torch.zeros(feature_num,feature_num)
    for i in range(n):
        current_x=x[i].reshape(feature_num,1)
        Si+=(current_x-mean) * (current_x-mean).T
    assert (Si.equal(Si.T))
    return Si

def compute_Sw(x_classified):
    Sw=torch.zeros(feature_num,feature_num)
    for i in range(10):
        Sw+=compute_Si(x_classified[i])
    return Sw

def compute_Sb(x_classified,train_x):
    x_head=meanX(train_x).reshape(feature_num,1)
    Sb=torch.zeros(feature_num,feature_num)
    for i in range(10):
        mean=meanX(x_classified[i]).reshape(feature_num,1)  # 784 -> 784X1
        n=np.shape(x_classified[i])[0]
        Sb+=n*(mean-x_head) * (mean-x_head).T
    return Sb

def LDA(x_classified, train_x):
    Sw=compute_Sw(x_classified)
    Sb=compute_Sb(x_classified,train_x)
    eig_value, vec = np.linalg.eig(np.linalg.pinv(Sw) * np.array(Sb))  # tensor -> numpy.ndarray
    index_vec = np.argsort(-eig_value)  # 对eig_value从大到小排序，返回索引
    eig_index = index_vec[:3]  # 取出最大的特征值的索引
    beta = vec[:, eig_index]  # 取出最大的特征值对应的特征向量
    return beta



train_batch=60000
test_batch=10000
feature_num=4
train_loader, test_loader=load_data()
x_classified, num, train_x, train_y, test_x, test_y = data_preprocess(train_loader,test_loader)
beta=LDA(x_classified,train_x)
newX1=[]
newX2=[]
newX3=[]
newY=[]
for i in range(test_batch):
    beta=beta.astype('float32')
    z1 = [np.matmul(test_x[i].T , beta)[0]]
    z2 = [np.matmul(test_x[i].T , beta)[1]]
    z3 = [np.matmul(test_x[i].T , beta)[2]]
    Z1 = np.sum(z1)
    Z2 = np.sum(z2)
    Z3 = np.sum(z3)
    label =[test_y[i]]
    Label=np.sum(label)
    # if Z1<-0.01 or Z1>0.01 or Z2>0.1 or Z2<-0.005:
    #     continue
    newX1.append(Z1)
    newX2.append(Z2)
    newX3.append(Z3)
    newY.append(Label)
print(newX1)
print(newX2)
print(newX3)
print(newY)


ax = plt.figure().add_subplot(111, projection = '3d')
area = np.pi * 4**2
ax.scatter(newX1, newX2, newX3, c=newY, s=area, alpha=0.4)
# plt.savefig('layer_{}_label'.format(layernum))
plt.show()
ax = plt.figure().add_subplot(111, projection='3d')
fig = plt.figure()

