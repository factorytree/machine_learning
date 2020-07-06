import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

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
        dataset=train_dataset, batch_size = batch_size
        )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size
    )
    return train_loader,test_loader

class GMM:
    def __init__(self,Data,K,weights = None,means = None,covars = None):
        """
        这是GMM（高斯混合模型）类的构造函数
        :param Data: 训练数据
        :param K: 高斯分布的个数
        :param weigths: 每个高斯分布的初始概率（权重）
        :param means: 高斯分布的均值向量
        :param covars: 高斯分布的协方差矩阵集合
        """
        self.Data = Data
        self.K = K
        if weights is not None:
            self.weights = weights
        else:
            self.weights  = np.random.rand(self.K)
            self.weights /= np.sum(self.weights)        # 归一化
        col = np.shape(self.Data)[1]
        if means is not None:
            self.means = means
        else:
            self.means = []
            for i in range(self.K):
                mean = np.random.rand(col)
                #mean = mean / np.sum(mean)        # 归一化
                self.means.append(mean)
        if covars is not None:
            self.covars = covars
        else:
            self.covars  = []
            for i in range(self.K):
                cov = np.random.rand(col,col)
                #cov = cov / np.sum(cov)                    # 归一化
                self.covars.append(cov)                     # cov是np.array,但是self.covars是list

    def Gaussian(self,x,mean,cov):
        """
        这是自定义的高斯分布概率密度函数
        :param x: 输入数据
        :param mean: 均值数组
        :param cov: 协方差矩阵
        :return: x的概率
        """
        dim = np.shape(cov)[0]
        # cov的行列式为零时的措施
        covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
        covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
        xdiff = (x - mean).reshape((1,dim))
        # 概率密度
        prob = 1.0/(np.power(np.power(2*np.pi,dim)*np.abs(covdet),0.5))*\
               np.exp(-0.5*xdiff.dot(covinv).dot(xdiff.T))[0][0]
        return prob

    def GMM_EM(self):
        """
        这是利用EM算法进行优化GMM参数的函数
        :return: 返回各组数据的属于每个分类的概率
        """
        loglikelyhood = 0
        oldloglikelyhood = 1
        len,dim = np.shape(self.Data)
        # gamma表示第n个样本属于第k个混合高斯的概率
        gammas = [np.zeros(self.K) for i in range(len)]
        while np.abs(loglikelyhood-oldloglikelyhood) > 0.00000001:
            oldloglikelyhood = loglikelyhood
            # E-step
            for n in range(len):
                # respons是GMM的EM算法中的权重w，即后验概率
                respons = [self.weights[k] * self.Gaussian(self.Data[n], self.means[k], self.covars[k])
                                                    for k in range(self.K)]
                respons = np.array(respons)
                sum_respons = np.sum(respons)
                gammas[n] = respons/sum_respons
            # M-step
            for k in range(self.K):
                #nk表示N个样本中有多少属于第k个高斯
                nk = np.sum([gammas[n][k] for n in range(len)])
                # 更新每个高斯分布的概率
                self.weights[k] = 1.0 * nk / len
                # 更新高斯分布的均值
                self.means[k] = (1.0/nk) * np.sum([gammas[n][k] * self.Data[n] for n in range(len)], axis=0)
                xdiffs = self.Data - self.means[k]
                # 更新高斯分布的协方差矩阵
                self.covars[k] = (1.0/nk)*np.sum([gammas[n][k]*xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)
            loglikelyhood = []
            for n in range(len):
                tmp = [np.sum(self.weights[k]*self.Gaussian(self.Data[n],self.means[k],self.covars[k])) for k in range(self.K)]
                tmp = np.log(np.array(tmp))
                loglikelyhood.append(list(tmp))
            loglikelyhood = np.sum(loglikelyhood)
        for i in range(len):
            gammas[i] = gammas[i]/np.sum(gammas[i])
        self.posibility = gammas
        self.prediction = [np.argmax(gammas[i]) for i in range(len)]

def get_label(filepath):
    file=open(filepath,'r')
    y=[]
    for line in file.readlines():
        line=line.rstrip('\n')
        y.append(int(line))
    file.close()
    return y

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

def run():
    # batch_size=10000
    # train_loader, test_loader = load_data(batch_size)
    # for batch_index, (x,y) in enumerate(test_loader):
    #     x=np.array(x)
    #     data = x.reshape(batch_size, 784)
    #     pca = PCA(n_components=4)  # 降到4维
    #     pca.fit(data)  # 训练
    #     data = pca.fit_transform(data)
    #     label=np.array(y)
    #     break
    filepath = 'LeNet/y2' # y1: 1000X400 y2: 1000X84
    data = load_tensor(filepath, pointnum=1000)
    print(np.shape(data))
    pca = PCA(n_components=4)  # 降到4维
    pca.fit(data)  # 训练
    data= pca.fit_transform(data)
    label=get_label('LeNet/label.txt')
    print("Mnist数据集的标签：\n",label)

    # 对数据进行预处理
    data = Normalizer().fit_transform(data)

    # 解决画图的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 数据可视化
    plt.scatter(data[:,0],data[:,1],c = label)
    plt.title("Mnist数据集显示")
    plt.show()

    # GMM模型
    K = 10
    gmm = GMM(data,K)
    gmm.GMM_EM()
    y_pre = gmm.prediction
    print("GMM预测结果：\n",y_pre)
    print("GMM正确率为：\n",accuracy_score(label,y_pre))
    plt.scatter(data[:, 0], data[:, 1], c=y_pre)
    plt.title("GMM结果显示")
    plt.show()


if __name__ == '__main__':
    run()