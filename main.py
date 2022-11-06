#!/usr/bin/env python
# coding: utf-8

# In[32]:


#解压数据集
#get_ipython().system('unzip -oq /home/aistudio/data/data175443/text_classification_data.zip')


# In[33]:


#读数据集的函数
def read_tsv(path):
    docs=[]
    labels=[]
    with open(path,'r') as f:
        for line in f.readlines():
            txt,label=line.split("\t")
            docs.append(txt)
            labels.append(label)
    return docs,labels


# In[34]:


#获得训练集和测试集
train_txt, train_labels=read_tsv('./text_classification_data/train.tsv')
test_txt, test_labels=read_tsv('./text_classification_data/test.tsv')


# In[35]:


#向量化
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()

X=vectorizer.fit_transform(train_txt+test_txt).toarray()
train_vec=X[:len(train_txt)]
test_vec=X[len(train_txt):]


# In[36]:


#训练模型
import numpy
from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
new_vec=train_vec
#print(numpy.asarray(train_vec))
#print(train_labels)
model.fit(train_vec,train_labels)
test_pre_labels=model.predict(test_vec)


# In[37]:


#保存模型
import joblib
joblib.dump(model,'./models/GaussianNB.pkl')


# In[38]:


#计算准确率和召回率
TP=0
TN=0
FN=0
FP=0
i=0
while i<len(test_labels):
    if(test_labels[i]=='1\n' and test_pre_labels[i]=='1\n'):
        TP+=1
    if(test_labels[i]=='0\n' and test_pre_labels[i]=='0\n'):
        TN+=1
    if(test_labels[i]=='1\n' and test_pre_labels[i]=='0\n'):
        FN+=1
    if(test_labels[i]=='0\n' and test_pre_labels[i]=='1\n'):
        FP+=1
    i+=1
print("召回率是"+str((TP+TN)/(TP+TN+FP+FN))+"\n")
print("准确率是"+str(TP/(TP+FN))+"\n")


# In[31]:


#实现实时分类
#print(len(test_vec[0]))
total_txt=test_txt+train_txt
while 1:
    print("请输入英文句子（不含标点符号）：")
    sim_txt=input()
    total_txt.append(sim_txt)
    total_vec=vectorizer.fit_transform(total_txt).toarray()
    #print(len(total_vec))
    while (len(test_vec[0])!=len(total_vec[0])):
        print(total_txt[len(total_txt)-1])
        total_txt.pop()
        print("你输入了语料库外的词语，请重新输入：")
        sim_txt=input()
        total_txt.append(sim_txt)
        total_vec=vectorizer.fit_transform(total_txt).toarray()
        #print(len(total_vec[0]))
    total_txt.reverse()
    total_vec=vectorizer.fit_transform(total_txt).toarray()
    print(total_txt[0])
    model1=joblib.load('./models/GaussianNB.pkl')
    print(model1.predict([total_vec[0]]))

   

