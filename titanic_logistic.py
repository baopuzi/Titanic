
# coding: utf-8
#导入数据
import numpy as np
import pandas as pd
#训练数据集
train=pd.read_csv('C:/Users/MQ/Desktop/DataSource/titanic/train.csv')
#测试数据集
test=pd.read_csv('C:/Users/MQ/Desktop/DataSource/titanic/test.csv')
print(train.shape)
print(test.shape)

#合并数据集，方便同时对两个数据集进行清洗
full=train.append(test,ignore_index=True)
full.shape
full.head()

#PassengerID乘客编号  Pclass客舱等级  SibSp同代直系亲属数  Parch不同代直系亲属数  Fare船票价格  Ticket船票编号  Cabin客舱号
#Embarked登船港口 出发地点S-英国南安普顿 途径地点C-法国瑟堡市 出发地点Q-爱尔兰昆士敦

full.info()

full.describe()


#处理缺失值  年龄和票价填充平均值
full['Age']=full['Age'].fillna(full['Age'].mean())
full['Fare']=full['Fare'].fillna(full['Fare'].mean())


#登船港口 缺失两条数据，用最频繁的值填充
full['Embarked'].value_counts()

full['Embarked']=full['Embarked'].fillna('S')

#船舱号，由于缺失数据比较多，填充为U，表示Uknow
full['Cabin']=full['Cabin'].fillna('U')

#特征工程：最大限度地从原始数据中提取特征，以供机器学习算法和模型使用
#特征提取：数值类型可以直接使用；时间序列转换成单独的年月日；分类数据用数值代替类别（One-hot编码）
#需要熟悉业务逻辑
#One-hot编码：如果原始数据某个特征有n个类别，则将这个特征扩充为对应的n个特征

#特征提取：性别
sex_mapDict={'male':1,'female':0}
full['Sex']=full['Sex'].map(sex_mapDict)
full.head()

#特征提取：客舱等级
pclassDf=pd.DataFrame()
pclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
pclassDf.head()

full=pd.concat([full,pclassDf],axis=1)
full.drop('Pclass',axis=1,inplace=True)
full.head()

#特征提取：登录港口
embarkedDf=pd.DataFrame()
embarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked')
embarkedDf.head()

full=pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
full.head()

full['Name'].head()

#特征提取：姓名
def getTitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3
#存放提取后的特征
titleDf=pd.DataFrame()
titleDf['title']=full['Name'].map(getTitle)
titleDf.head()

titleDf['title'].value_counts()

title_mapDict={
    'Mr':'Mr',              
    'Miss':'Miss',            
    'Mrs': 'Mrs',           
    'Master':'Master',        
    'Dr':'Officer',               
    'Rev':'Officer',               
    'Col':'Officer',               
    'Mlle':'Miss',              
    'Ms':'Mrs',                
    'Major':'Officer',             
    'Dona':'Royalty',              
    'Jonkheer':'Royalty',          
    'Mme':'Mrs',               
    'the Countess':'Royalty',      
    'Capt':'Officer',              
    'Lady':'Royalty',              
    'Don':'Royalty',             
    'Sir':'Royalty'
}
titleDf['title']=titleDf['title'].map(title_mapDict)
titleDf=pd.get_dummies(titleDf['title'])
titleDf.head()

full=pd.concat([full,titleDf],axis=1)
full.drop('Name',axis=1,inplace=True)
full.head()

#特征提取：客舱号
full['Cabin'].head()


#存放客舱号信息
cabinDf=pd.DataFrame()
full['Cabin']=full['Cabin'].map(lambda Cab:Cab[0])
cabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')
cabinDf.head()

full=pd.concat([full,cabinDf],axis=1)
full.drop('Cabin',axis=1,inplace=True)
full.head()


#特征提取：家庭类别
familyDf=pd.DataFrame()
familyDf['familySize']=full['Parch']+full['SibSp']+1
familyDf['Family_Simple']=familyDf['familySize'].map(lambda S:1 if S==1 else 0)
familyDf['Family_Small']=familyDf['familySize'].map(lambda S:1 if 1<S<5 else 0)
familyDf['Family_Larger']=familyDf['familySize'].map(lambda S:1 if S>4 else 0)
familyDf.head()

full=pd.concat([full,familyDf],axis=1)
full.head()


full.head()


full.shape


#相关系数
corrDf=full.corr()
corrDf


corrDf['Survived'].sort_values(ascending=False)


#特征选择
full_X=pd.concat([titleDf,
                 pclassDf,
                 familyDf,
                 full['Fare'],
                 cabinDf,
                 embarkedDf,
                 full['Sex']]
                 ,axis=1
                )
full_X.head()



#原始数据集有891
sourceRow=891
#原始数据集：特征
source_X=full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y=full.loc[0:sourceRow-1,'Survived']

#预测数据集特征
pred_X=full_X.loc[sourceRow:,:]
pred_X.shape

from sklearn.model_selection import train_test_split
#建立模型用的训练数据集和测试数据集
train_X,test_X,train_y,test_y=train_test_split(source_X,source_y,train_size=0.8,test_size=0.2)
print('原始数据集特征：',source_X.shape,'\n','训练数据集特征：',train_X.shape,'\n','测试数据集特征：',test_X.shape,'\r')
print('原始数据集标签：',source_y.shape,'\n','训练数据集标签：',train_y.shape,'\n','测试数据集标签：',test_y.shape,'\r')

from sklearn.linear_model import  LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(train_X,train_y)
model.score(test_X,test_y)

#方案实施：使用机器学习模型，对预测数据中的生存情况进行预测
pred_y=model.predict(pred_X)

pred_y=pred_y.astype(int)

passenger_id=full.loc[sourceRow:,'PassengerId']
predDf=pd.DataFrame({'PassengerId':passenger_id,
                   'Survived':pred_y})
predDf.shape
predDf.head()

