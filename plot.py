import matplotlib.pyplot as plt

root='1_project/logistic_regression_Ridge/'
path=['lr_ridge_lambda_1e-4_0.01','lr_ridge_lambda_1e-4_0.9','lr_ridge_lambda_1e-4_1e-4','lr_ridge_lambda_1e-4_5','lr_ridge_lambda_1e-4_100',
      'lr_ridge_lambda_1e-05_0.1','lr_ridge_lambda_1e-05_0.01','lr_ridge_lambda_1e-05_0.5','lr_ridge_lambda_1e-05_1','lr_ridge_lambda_1e-05_2']

train_acc=[]
test_acc=[]
train_loss=[]
test_loss=[]
epoch=[]
for i in range(1):
    file=open('1_project/LeNet/LeNet_lr_0.001_MSELoss.txt','r')
    line=file.readline()
    for line in file.readlines():
        line=line.rstrip('\n')
        line=line.split()
        line=[float(line[i]) for i in range(len(line))]
        epoch.append(line[0])
        train_acc.append(line[1])
        test_acc.append(line[2])
        train_loss.append(line[3])
        test_loss.append(line[4])
    file.close()

def acc_graph2():
    Lambda=['0.01','0.9','1e-4','5','100','0.1','0.01','0.5','1','2']
    color=['blue','skyblue','pink','red','lightgreen','blue','skyblue','pink','red','lightgreen']
    for i in range(5,10):
        plt.plot(epoch[i], train_acc[i], color=color[i],label='lambda='+Lambda[i])
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Train Accuracy')

    plt.savefig('acc_1e-5.png')
    plt.show()

def acc_graph():
    plt.plot(epoch, train_acc,color='pink',label='Train Accuracy')
    plt.plot(epoch,test_acc,color='red',label='Test Accuracy')

    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')

    plt.savefig('acc_mse.png')
    plt.show()

def loss_graph():
    plt.plot(epoch, train_loss, color='blue', label='Train Loss')
    plt.plot(epoch, test_loss, color='skyblue', label='Test Loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')

    plt.savefig('loss_mse.png')
    plt.show()


#acc_graph2()
#loss_graph()
acc_graph()