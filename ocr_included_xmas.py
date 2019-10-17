#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xlrd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc,  accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from xgboost import XGBClassifier
from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
import random
import requests
import json
import base64
import urllib
import sys
import ssl
import graphviz
import time


# In[ ]:


def readxls(root):
    data_list=[]
    data=xlrd.open_workbook(root)
    table=data.sheets()[0]
    nrows=table.nrows
    ncols=table.ncols
    for i in range(1,nrows):
        data_list.append(table.row_values(i))
    rowname=table.row_values(0)
    return data_list,rowname


# In[ ]:


healthy,rowname=readxls("negative_data.xls")
unhealthy,rowname=readxls("positive_data.xls")
total_data=healthy+unhealthy
total_data=DataFrame(total_data)
total_data.columns=rowname
#print(total_data)
target=[0]*len(healthy)+[1]*len(unhealthy)
X_train, X_test, y_train, y_test =train_test_split(total_data, target, test_size=0.25, random_state=99999)


# In[ ]:


start = time.clock()
clf = ExtraTreesClassifier()
clf = clf.fit(X_train, y_train)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True,threshold=0.03)
X_train = model.transform(X_train)
X_test=model.transform(X_test)
print(len(X_train[0]))
end = time.clock()
print(str(end-start))


# In[ ]:


select_result=DataFrame(clf.feature_importances_).sort_values(by=0).T
select_result.columns=rowname
select_result.to_csv('feature selection.csv')
print(select_result)


# In[ ]:


sel_rows=np.array(rowname)[clf.feature_importances_>=0.03]


# In[ ]:


X_train=DataFrame(X_train)
X_train.columns=np.array(rowname)[clf.feature_importances_>=0.03]
X_test=DataFrame(X_test)
X_test.columns=np.array(rowname)[clf.feature_importances_>=0.03]


# In[ ]:


start=time.clock()
clf = tree.DecisionTreeClassifier(max_depth=6,min_samples_split=12)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
end = time.clock()
print(str(end-start))


# In[ ]:


dot_tree = tree.export_graphviz(clf,out_file=None,feature_names=sel_rows,class_names=['未得肾病','得肾病'],filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_tree)
graph.write_png("tree6.png")


# In[ ]:


y_score = clf.fit(X_train, y_train).predict_proba(X_test)
y_score=[a[1] for a in y_score]
#y_pred=y_score>=threshold
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


clf.feature_importances_


# In[ ]:


start=time.clock()
svclassifier = SVC(kernel='poly',degree=3,class_weight={1:len(unhealthy)/len(target),0:len(healthy)/len(target)},probability=True)
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
end = time.clock()
print(str(end-start))


# In[ ]:


print(svclassifier)


# In[ ]:


y_score=svclassifier.fit(X_train, y_train).decision_function(X_test)

fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive  Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


start = time.clock()
model = XGBClassifier(booster='gbtree',max_depth=5,eval_metric='auc',learning_rate=0.7,min_child_weight= 0.9,verbose_eval=True)
model.fit(DataFrame(X_train,dtype='float'), DataFrame(y_train))
end = time.clock()
print('训练模型的时间为'+str(end-start))
start = time.clock()
y_pred = model.predict(DataFrame(X_test,dtype='float'))
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test,predictions))
end = time.clock()
print('预测163个结果，所需时间为'+str(end-start))


# In[ ]:


print(model)


# In[ ]:


xgb.to_graphviz(model, num_trees=10)


# In[ ]:


y_score=model.predict_proba(DataFrame(X_test,dtype='float'))
y_score=[a[1] for a in y_score]
fpr,tpr,threshold = roc_curve(y_test, y_score) ###计算真正率和假正率
roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#从百度api调用ocr自定义模板识别
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=ppPj2gyINjoYiqkhsjAnyYDC&client_secret=2Q6tsZrbGsE60pXuoxg5o5AOUDCSMaLP'
header = { 'Content-Type':'application/json; charset=UTF-8' }
r = requests.post(host,headers = header)
r = json.loads(r.text)
Access_token = r['access_token']

f = open(r'test.jpg', 'rb')
img = base64.b64encode(f.read())
data = {"image": img,"templateSign":"7dc32854acac2c3bac8d3bb599ceaeca"}
ocr_host = 'https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise?access_token='+Access_token
ocr_header = {'Content-Type':'application/x-www-form-urlencoded',"apikey":"ppPj2gyINjoYiqkhsjAnyYDC"}
img =  requests.post(ocr_host,headers = ocr_header,data=data )
img= json.loads(img.text)
print(img["data"]["ret"])
#可以得到返回的结果如下所示，我们需要将其整理成我们可以利用的形式


# In[ ]:


ocr_res=img["data"]["ret"]
sim_res=[i['word']  for i in ocr_res]
sim_res
#print(img["data"]["ret"][0]['word'])


# In[ ]:


testdata=DataFrame(sim_res[1::2]).T
testdata.columns=sim_res[0::2]
testdata


# In[ ]:


testdata.columns
testdata=testdata.rename(columns={'中性细胞比率': '中性粒细胞百分比',
                                  '淋巴细胞(%)':'淋巴细胞百分比', 
                                  '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                  '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                  '中性细胞数':'中性粒细胞计数',
                                  '淋巴细胞值':'淋巴细胞数计数',
                                  '单核细胞百分比':'单核细胞',
                                  '嗜酸性粒细胞':'嗜酸性粒细胞计数',
                                  '嗜碱性粒细胞':'嗜碱性粒细胞计数',
                                  '红细胞平均体积':'平均红细胞体积',
                                  '平均血红蛋白量':'平均血红蛋白',
                                  '红细胞分布宽度':'红细胞分布宽度变异系数',
                                  '平均血小板体积':'血小板平均体积',
                                  '血小板分布宽度':'血小板平均分布宽度'})


# In[ ]:


testdata=testdata.apply(pd.to_numeric, errors='ignore')


# In[ ]:


xtest=testdata[np.array(rowname)[clf.feature_importances_>=0.03]]


# In[ ]:


model.predict(xtest)


# In[ ]:


#从百度api调用ocr自定义模板识别
start=end = time.clock()
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=ppPj2gyINjoYiqkhsjAnyYDC&client_secret=2Q6tsZrbGsE60pXuoxg5o5AOUDCSMaLP'
header = { 'Content-Type':'application/json; charset=UTF-8' }
r = requests.post(host,headers = header)
r = json.loads(r.text)
Access_token = r['access_token']

f = open(r'test.jpg', 'rb')
img = base64.b64encode(f.read())
data = {"image": img,"templateSign":"7dc32854acac2c3bac8d3bb599ceaeca"}
ocr_host = 'https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise?access_token='+Access_token
ocr_header = {'Content-Type':'application/x-www-form-urlencoded',"apikey":"ppPj2gyINjoYiqkhsjAnyYDC"}
img =  requests.post(ocr_host,headers = ocr_header,data=data )
img= json.loads(img.text)
ocr_res=img["data"]["ret"]
sim_res=[i['word']  for i in ocr_res]
testdata=DataFrame(sim_res[1::2]).T
testdata.columns=sim_res[0::2]
testdata.columns
testdata=testdata.rename(columns={'中性细胞比率': '中性粒细胞百分比',
                                  '淋巴细胞(%)':'淋巴细胞百分比', 
                                  '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                  '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                  '中性细胞数':'中性粒细胞计数',
                                  '淋巴细胞值':'淋巴细胞数计数',
                                  '单核细胞百分比':'单核细胞',
                                  '嗜酸性粒细胞':'嗜酸性粒细胞计数',
                                  '嗜碱性粒细胞':'嗜碱性粒细胞计数',
                                  '红细胞平均体积':'平均红细胞体积',
                                  '平均血红蛋白量':'平均血红蛋白',
                                  '红细胞分布宽度':'红细胞分布宽度变异系数',
                                  '平均血小板体积':'血小板平均体积',
                                  '血小板分布宽度':'血小板平均分布宽度'})
testdata=testdata.apply(pd.to_numeric, errors='ignore')
xtest=testdata[np.array(rowname)[clf.feature_importances_>=0.03]]
#print(xtest)
prob=model.predict_proba(xtest).tolist()[0]
if model.predict(xtest):
    print('该人得有肾病,概率为%f'%prob[1])
else:
        print('该人未得肾病，概率为%f'%prob[0])
end = time.clock()
print('运行时间为'+str(end-start)+'秒')


# In[ ]:


len(X_test)


# In[ ]:


def bingxing(filename):
    start=end = time.clock()
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=ppPj2gyINjoYiqkhsjAnyYDC&client_secret=2Q6tsZrbGsE60pXuoxg5o5AOUDCSMaLP'
    header = { 'Content-Type':'application/json; charset=UTF-8' }
    r = requests.post(host,headers = header)
    r = json.loads(r.text)
    Access_token = r['access_token']

    f = open(filename, 'rb')
    img = base64.b64encode(f.read())
    data = {"image": img,"templateSign":"7dc32854acac2c3bac8d3bb599ceaeca"}
    ocr_host = 'https://aip.baidubce.com/rest/2.0/solution/v1/iocr/recognise?access_token='+Access_token
    ocr_header = {'Content-Type':'application/x-www-form-urlencoded',"apikey":"ppPj2gyINjoYiqkhsjAnyYDC"}
    img =  requests.post(ocr_host,headers = ocr_header,data=data )
    img= json.loads(img.text)
    ocr_res=img["data"]["ret"]
    sim_res=[i['word']  for i in ocr_res]
    testdata=DataFrame(sim_res[1::2]).T
    testdata.columns=sim_res[0::2]
    testdata.columns
    testdata=testdata.rename(columns={'中性细胞比率': '中性粒细胞百分比',
                                      '淋巴细胞(%)':'淋巴细胞百分比', 
                                      '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                      '嗜酸性粒细胞比': '嗜酸性粒细胞百分比',
                                      '中性细胞数':'中性粒细胞计数',
                                      '淋巴细胞值':'淋巴细胞数计数',
                                      '单核细胞百分比':'单核细胞',
                                      '嗜酸性粒细胞':'嗜酸性粒细胞计数',
                                      '嗜碱性粒细胞':'嗜碱性粒细胞计数',
                                      '红细胞平均体积':'平均红细胞体积',
                                      '平均血红蛋白量':'平均血红蛋白',
                                      '红细胞分布宽度':'红细胞分布宽度变异系数',
                                      '平均血小板体积':'血小板平均体积',
                                      '血小板分布宽度':'血小板平均分布宽度'})
    testdata=testdata.apply(pd.to_numeric, errors='ignore')
    xtest=testdata[np.array(rowname)[clf.feature_importances_>=0.03]]
    #print(xtest)
    prob=model.predict_proba(xtest).tolist()[0]
    if model.predict(xtest):
        print('该人得有肾病,概率为%f'%prob[1])
    else:
            print('该人未得肾病，概率为%f'%prob[0])
    end = time.clock()
    print('运行时间为'+str(end-start)+'秒')


# In[ ]:


from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(20)
filelist=['test.jpg','1.jpg',...,]
pool.map(bingxing,filelist)
pool.close()
pool.join()   

