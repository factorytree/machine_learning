import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

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

def get_size(train_loader):
    for batch_index, (x, y) in enumerate(train_loader):
        pic_num, _, pic_size, _ = np.shape(x)
        feature_num = pic_size ** 2
        break
    return pic_num, pic_size, feature_num

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def train(beta):
    total_pic = 0
    match = 0
    loss = 0
    for batch_index, (x,y) in enumerate(train_loader):
        x=x.reshape(pic_num,feature_num) # 100X1X28X28 -> 100X784
        y=y.reshape(pic_num,1)
        y_onehot=torch.zeros(pic_num, 10).scatter_(1, y, 1) # 100X1 -> 100X10
        y_linear=np.matmul(x,beta)
        y_star=sigmoid(y_linear) # 100X10

        error=y_onehot-y_star # 100X10
        beta=beta+learning_rate*(np.matmul(x.T,error))
        Loss=[-np.dot(y_onehot[i],y_linear[i])+torch.sum(np.log(torch.ones(10)+np.exp(y_linear[i]))) for i in range(pic_num)]
        loss+=np.sum(Loss)

        y_star = np.argmax(y_star, axis=1)

        total_pic += pic_num
        for i in range(pic_num):
            if y_star[i] == y[i]:
                match += 1
    return beta, match/total_pic, loss/60000

def test(beta):
    total_pic=0
    match=0
    loss=0
    for batch_index, (x,y) in enumerate(test_loader):
        x=x.reshape(pic_num,feature_num) # 100X1X28X28 -> 100X784
        y = y.reshape(pic_num, 1)
        y_onehot = torch.zeros(pic_num, 10).scatter_(1, y, 1)  # 100X1 -> 100X10
        y_star=sigmoid(np.matmul(x,beta)) # 100X10
        y_linear = np.matmul(x, beta)

        Loss = [-np.dot(y_onehot[i], y_linear[i]) + torch.sum(np.log(torch.ones(10) + np.exp(y_linear[i]))) for i in
                range(pic_num)]
        loss += np.sum(Loss)

        y_star=np.argmax(y_star,axis=1)
        total_pic+=pic_num
        for i in range(pic_num):
            if y_star[i]==y[i]:
                match+=1
    return match/total_pic, loss/10000

train_loader, test_loader = load_data(batch_size=100)
pic_num, pic_size, feature_num = get_size(train_loader) # 100, 28, 784
learning_rate=1e-4
iter=200
beta=torch.rand(feature_num,10) # 784X10
file=open('lr.txt_{}'.format(str(learning_rate)),'w')
for i in range(iter):
    beta, train_acc, train_loss=train(beta)
    test_acc, test_loss=test(beta)
    print(i,train_acc,test_acc,train_loss,test_loss)
    file.write(str(i)+'\t'+str(train_acc)+'\t'+str(test_acc)+
               '\t'+str(train_loss)+'\t'+str(test_loss)+'\n')
file.close()
