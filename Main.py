from sklearn import datasets as ds
import numpy as np
    
def loss(X,Y,theta):#计算X在h函数的结果y中的全部损失
    num=np.size(Y)
    h=1/(1+np.exp(-np.dot(X,theta)))
    Y=np.transpose(Y)
    loss=(-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h))))/num
    return loss[0][0]

def der(X,Y,theta):
    num=np.size(Y)
    h=1/(1+np.exp(-np.dot(X,theta)))
    temp=h-Y
    return np.dot(X.transpose(),temp)/num
    
def train(Xtrain,Ytrain,Xvalidation,Yvalidationtemp):           
    #设置初始参数
    theta=np.zeros((5,1))
    learnRatio=2#定义学习率
    times=20000#定义循环次数
    
    #开始训练
    for i in range(0,times):
        if i%1000==0:
            losstrain=loss(Xtrain,Ytrain,theta)
            lossvalidation=loss(Xvalidation,Yvalidationtemp,theta)
            print("After %d\t,loss in train is %.7f\t loss in validation is %.7f"%(i,losstrain,lossvalidation))
        theta=theta-learnRatio*der(Xtrain,Ytrain,theta)  
    theta=theta.reshape(5,)
    return theta
    

if __name__=='__main__':
    print("---start---")
    
    #读入数据集
    iris = ds.load_iris()
    X = iris.data#4维属性 150个数据集
    Y = iris.target#结果为整数，已经按50一类排好序
    Y=Y.reshape(150,1)#左列为1的增广
    xx=np.ones((150,1))
    X = np.c_[xx,X]#5维属性 150个数据集,左列为1

    #对数据进行特征缩放
    for i in range(1,5):
        maxn=X[:,i].max()
        minn=X[:,i].min()
        meann=X[:,i].mean()
        X[:,i]=(X[:,i]-meann)/(maxn-minn)  
    
    #划分训练集与测试集,对于每一个分类0-30训练集 30-40验证集 40-50测试集
    Xtrain=np.vstack((X[0:30],X[50:80],X[100:130]))#前40个作为训练集
    Ytrain=np.vstack((Y[0:30],Y[50:80],Y[100:130]))
    Xvalidation=np.vstack((X[30:40],X[80:90],X[130:140]))#后10个作为测试集
    Yvalidation=np.vstack((Y[30:40],Y[80:90],Y[130:140]))
    XTest=np.vstack((X[40:50],X[90:100],X[140:150]))
    YTest=np.vstack((Y[40:50],Y[90:100],Y[140:150]))
    
    #求训练参数
    theta=np.zeros((3,5))
    for i in range(0,3):
        #对训练集标签进行修正
        print("---start train: "+str(i))
        Ytraintemp=Ytrain.copy()
        Yvalidationtemp=Yvalidation.copy()
        for j in range(0,90):
            if Ytraintemp[j,0]==i:
                Ytraintemp[j,0]=1
            else:
                Ytraintemp[j,0]=0
        #对验证集进行标签修正
        for j in range(0,30):
            if Yvalidationtemp[j,0]==i:
                Yvalidationtemp[j,0]=1
            else:
                Yvalidationtemp[j,0]=0
          
        #求theta参数
        theta[i]=train(Xtrain,Ytraintemp,Xvalidation,Yvalidationtemp)
        
    #利用测试集进行测试
    theta=theta.transpose()
    h3=1/(1+np.exp(-np.dot(XTest,theta)))
    for i in range(0,30):
        pre=np.argmax(h3[i])
        print("predict:"+str(pre)+" real:"+str(YTest[i,0]))        
    print("---end---")