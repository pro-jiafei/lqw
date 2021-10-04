import pandas as pd
data1 = pd.read_csv('testData.csv')
print(len(data1))
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn.tree import DecisionTreeClassifier

data1.columns = ['虫口密度','发生程度等级','均日降水量(30天)', '平均气压（30天）', '平均风速（30天）', '平均温度（30天）', '平均水汽压（30天）', '平均相对湿度（30天）', '平均2020时日照时数（30天）',
                 '平均气压日较差（30天）', '平均最大风速（30天）', '平均气温日较差（30天）', '33℃以上及-8℃以下温度天数（30天）', '平均日降水量（60天）', '平均气压（60天）',
                 '平均风速（60天）', '平均温度（60天）', '平均水汽压（60天）', '平均相对湿度（60天）', '平均2020时日照时数（60天）', '平均气压日较差（60天）',
                 '平均最大风速（60天）', '平均气温日较差（60天）', '33℃以上及-8℃以下温度天数（60天）', '平均日降水量（90天）', '平均气压（90天）', '平均风速（90天）',
                 '平均温度（90天）', '平均水汽压（90天）', '平均相对湿度（90天）', '平均2020时日照时数（90天）', '平均气压日较差（90天）', '平均最大风速(90天)',
                 '33℃以上及-8℃以下温度天数（90天)','平均最大风速（90天)']
print(data1)
from sklearn.model_selection import train_test_split
print(type(data1))
x,y=data1.iloc[:,1:].values,data1.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
feat_labels=data1.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(x_train,y_train)
#n_estimators:森林中树的数量
#n_jobs   整数可选（默认=1）适合和预测并运行的作业数，如果为-1,则将作业数设置为核心数


#下面对训练好的随机森林，完成重要性评估
#feature_importances_   可以调取关于特征重要程度
importances=forest.feature_importances_
print("重要性：",importances)
x_colummns=data1.columns[1:]
indices=np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
threshold=0.15
x_selected=x_train[:,importances > threshold]
#可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.title("影响因子的重要程度",fontsize=18)
plt.ylabel("import level",fontsize=15,rotation=90)
plt.rcParams['font.sans-serif']=["SimHei"]
plt.rcParams['axes.unicode_minus']=False
for i in range(x_colummns.shape[0]):
    plt.bar(i,importances[indices[i]],color='orange',align='center')
    plt.xticks(np.arange(x_colummns.shape[0]),x_colummns, rotation=90, fontsize=15)
plt.show()
print(plt)
