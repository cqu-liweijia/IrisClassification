from sklearn import datasets as ds
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__=='__main__':
    print("---start---")
    
    #读入数据集
    iris = ds.load_iris()
    X = iris.data#4维属性 150个数据集
    Y = iris.target#结果为整数，已经按50一类排好序
    Y=Y.reshape(150,1)
    
    Xtrain=np.vstack((X[0:40],X[50:90],X[100:140]))#前40个作为训练集
    Ytrain=np.vstack((Y[0:40],Y[50:90],Y[100:140]))
    XTest=np.vstack((X[40:50],X[90:100],X[140:150]))#后10个作为测试集
    YTest=np.vstack((Y[40:50],Y[90:100],Y[140:150]))
    
    #引用模型并训练
    lr = LogisticRegression(solver='liblinear',multi_class='ovr')
    lr.fit(Xtrain,Ytrain)

    #输出结果
    result = lr.predict(XTest)
    for i in range(0,30):
        print("real:"+str(result[i])+" predict:"+str(result[i]))
    
    print("---end---")