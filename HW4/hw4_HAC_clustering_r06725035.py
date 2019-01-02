#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import wordpunct_tokenize
# from nltk import word_tokenize
# from nltk.stem.porter import *
# from nltk.tokenize import RegexpTokenizer
import os,sys
# import re
import pickle
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
# import random
# import math
# from collections import Counter
# import functools
import heapq
# import time , heapq
# from itertools import * #Functions creating iterators for efficient looping


# In[2]:


def cosine(DOCx, DOCy):
    '''
    input: doc1 path name(str) , doc2 path name(str)
    ouput: two doc's cosine similarity
    '''
    dfx = pd.read_csv(DOCx,sep=' ',names=['tindex','tfidf'],header=None)
    dfy = pd.read_csv(DOCy,sep=' ',names=['tindex','tfidf'],header=None)
    dfx = dfx.drop(0)
    dfy = dfy.drop(0)
    dfxy = pd.merge(dfx,dfy,on='tindex',how='outer')
    dfxy.fillna(0,inplace=True)
    up = sum(dfxy.tfidf_x * dfxy.tfidf_y)
    down = np.sqrt(sum(np.square(dfxy.tfidf_x)))* np.sqrt(sum(np.square(dfxy.tfidf_y)))
    result = up / down
    return result


# similarity measure between clusters

# In[3]:


def merge_sim(C, j, i, m):
    a = np.max([C[j][i], C[j][m]]) #single link
    b = np.min([C[j][i], C[j][m]]) # complete link
    return (a+b)/2 # max: single，min:complete


# In[4]:


def heap_merge_sim(C, j, i, m):
    a = np.min([C[j][i][0], C[j][m][0]])
    b = np.max([C[j][i][0], C[j][m][0]])
    return (a+b)/2#max 取不相似的 #min:取相似的(single)


# Cosine similarity for pair-wise document similarity

# In[5]:


# N=1095
# C=np.zeros([N,N])
# I=np.ones((N,), dtype=int)
# eps=1e-10
# for i in tqdm(range(N)):
#     for j in range(N-i-1):
#         cos = cosine('data/tf-idf/'+str(i+1)+'.txt','data/tf-idf/'+str(j+i+2)+'.txt')
#         C[i][j+i+1]=C[j+i+1][i]=cos+eps
# print(C.shape)
# pickle.dump(obj=C,file=open('data/C.pkl','wb'))


# In[6]:


N=1095
I=np.ones((N,), dtype=int)
eps=1e-10

cluster_representations = []
simple_cluster_results = []
C = pickle.load(open('data/C.pkl','rb'))
Ks = [8, 13, 20]
for n in range(N):
    cluster_representations.append([n])


# original HAC

# In[7]:


# simple_HAC_stime = time.time()
simple_A = []
for k in tqdm(range(N - 1)):

    max_sim = 0
    max_i = 0
    max_m = 0
    for i in range(N): #去找最大相似度的df<i,m><-argmax
        for m in range(i + 1):
            if i != m and I[i] == 1 and I[m] == 1 and C[i][m] >= max_sim:
                max_sim = C[i][m]
                max_i = i
                max_m = m
                
    simple_A.append((max_i, max_m))

    cluster_representations[max_i] += cluster_representations[max_m] #把m並進去i
    cluster_representations[max_m] = None
    
    for j in range(N): #更新全部文章對i的similarity
        the_sim = merge_sim(C, j, max_i, max_m)
        C[max_i][j] = the_sim
        C[j][max_i] = the_sim
        
    I[max_m] = 0 #m已經合併
    
    if np.sum(I) in Ks: #如果現在還活著的個數有再想要的分群當中

        the_cluster_result = sorted([sorted(cluster) for cluster in cluster_representations if cluster is not None])
        simple_cluster_results.append(the_cluster_result)


# print('Simple HAC Time Taken:', time.time() - simple_HAC_stime)
simple_cluster_results


# HEAP HAC
# * 只需要simple的三分之一時間

# In[8]:


heap_cluster_results = []
C_ = pickle.load(open('data/C.pkl','rb'))
C = []
P = []
I = np.zeros(N)
cluster_representations = []
N=1095
# heap_HAC_stime = time.time()
for n in range(N):
    c = []
    for i in range(N):
        the_sim = -C_[n][i]     
        c.append([the_sim, i])
    C.append(c)
    I[n] = 1
    C_list = sorted(C[n])
    C_list.remove(C[n][n])
    P.append(C_list)
    heapq.heapify(P[n])
    cluster_representations.append([n])
heap_A = [] # list of merges
for k in tqdm(range(N - 1)):
    min_neg_sim = 0
    min_i = 0
    min_m = 0
    for i in range(N):
        if I[i] == 1 and P[i][0][0] <= min_neg_sim: #similarity 最大的
            min_i = i
            min_neg_sim = P[i][0][0]
    min_m = P[min_i][0][1]
    k2, k1 = sorted([min_i, min_m]) 
    heap_A.append((k1, k2))
    I[k2] = 0 #k2併去k1
    P[k1] = []
    for i in range(N):
        if I[i] == 1 and i != k1: #for each i with 
            P[i].remove(C[i][k1])
            P[i].remove(C[i][k2])
            heapq.heapify(P[i])
            the_sim = heap_merge_sim(C, i, k1, k2) #sim
            C[i][k1][0] = the_sim
            heapq.heappush(P[i], C[i][k1])
            C[k1][i][0] = the_sim
            heapq.heappush(P[k1], C[k1][i])
    cluster_representations[k1] += cluster_representations[k2]
    cluster_representations[k2] = None
    if np.sum(I) in Ks:
        the_cluster_result = sorted([sorted(cluster) for cluster in cluster_representations if cluster is not None])
        heap_cluster_results.append(the_cluster_result)
# print('Heap HAC_ Time Taken', time.time() - heap_HAC_stime)
heap_cluster_results


# In[9]:


82.96472764015198 / 250.30226469039917


# 各種不同similarity link算法

# In[10]:


simple_single = simple_cluster_results
heap_single = heap_cluster_results
len(simple_single[2]) , len(heap_single[2])


# In[11]:


simple_avg1 = simple_cluster_results #幾何平均數
heap_avg1 = heap_cluster_results
len(simple_avg1[2]) , len(heap_avg1[2])


# In[12]:


simple_avg2 = simple_cluster_results #算術平均數
heap_avg2 = heap_cluster_results
len(simple_avg2[2]) , len(heap_avg2[2])


# In[13]:


simple_complete = simple_cluster_results
heap_complete = heap_cluster_results
len(simple_complete[2]) , len(heap_complete[2])


# In[14]:


with open('./data/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_dict[trainid[0]] = trainid[1:]
train_dict #class:doc_id
# train_dict = pickle.load(open('data/train_dict.pkl','rb'))


# In[15]:


df1 = pd.DataFrame.from_dict(train_dict)
df1 = df1.astype('int')
df1['2'].tolist()


# In[16]:


df = pd.read_csv('data/voting_rev.csv')
clf13=[]
for i in range(13):
    
    temp = df[df.Value == (i+1)]['id'].tolist()
    temp.extend(df1[str(i+1)].tolist())
    temp = sorted(temp)
    clf13.append(temp)
len(clf13)


# voting

# In[17]:


# pickle.dump(obj=voting_lis , file=open('data/voting_lis.pkl','wb'))
# pickle.dump(obj=clf13 , file=open('data/clf13.pkl','wb'))
voting_lis = pickle.load(open('data/voting_lis.pkl','rb'))
clf13 = pickle.load(open('data/clf13.pkl','rb'))


# In[18]:


# voting_li8 = [ans_all , ans_all,ans_all,ans_all,ans_all,ans_all,ans_all,ans_all,ans_all,ans_all,
#              simple_complete,simple_complete,simple_complete,heap_complete,heap_complete,heap_complete,
#              simple_single ,simple_avg1,heap_avg1,simple_avg2 ,heap_avg2] #=>8
# voting_li20 = [ans_all , ans_all,simple_complete,simple_complete,simple_complete,simple_complete,
#              heap_complete,heap_complete,heap_complete,heap_complete,simple_single ,
#              simple_avg1   ,heap_avg1,simple_avg2,heap_avg2 ] #20
# voting_li13 = [ans_all ,simple_complete,heap_complete,simple_single ,
#              simple_avg1,heap_avg1,simple_avg2, heap_avg2] #13
# voting_lis = [voting_li20,voting_li13,voting_li8]


# In[19]:


# voting_li = [ans_all , ans_all , simple_complete , heap_complete , simple_single , 
#              simple_avg1 ,heap_avg1, simple_avg2 ] #3154/1613/4332
# voting_li = [ans_all , ans_all , ans_all,simple_complete , heap_complete , simple_single , 
#              simple_avg1 ,heap_avg1, simple_avg2 ] #2073/2073/3929
# voting_li = [ans_all  ,simple_complete , heap_complete , simple_single , 
#              simple_avg1 ,heap_avg1, simple_avg2 ] #/2095/3871
# voting_li = [ans_all,ans_all  ,simple_complete , heap_complete , simple_single , heap_single,
#              simple_avg1 ,heap_avg1, simple_avg2,heap_avg2 ] #/1777/3730
# voting_li = [ans_all,ans_all  ,simple_complete , heap_complete ,
#              simple_avg1 ,heap_avg1, simple_avg2,heap_avg2 ] #/1359/4221
# voting_li = [ans_all , ans_all , ans_all,simple_complete , heap_complete , simple_single , 
#              simple_avg1 ,heap_avg1, simple_avg2,heap_avg2 ] #3215/1449/4243
# voting_li = [ans_all , ans_all ,simple_complete , heap_complete , simple_single , 
#              simple_avg1 ,heap_avg1, simple_avg2,heap_avg2 ] #1925/1925/4046
# voting_li = [ans_all , ans_all, ans_all, ans_all, ans_all, ans_all, ans_all, 
#              simple_complete ,simple_complete ,simple_complete ,simple_complete ,
#              heap_complete ,heap_complete ,heap_complete ,heap_complete ,
#              simple_single ,
#              simple_avg1   ,heap_avg1,
#              simple_avg2 ] #2230/3515=>578/1175
# voting_li = [ans_all , ans_all, ans_all, ans_all, ans_all, ans_all, 
#              simple_complete ,simple_complete ,simple_complete ,simple_complete ,
#              heap_complete ,heap_complete ,heap_complete ,heap_complete ,
#              simple_single ,
#              simple_avg1   ,heap_avg1,
#              simple_avg2 ] #2226/3417=>20:574/1105
# voting_li = [ans_all , ans_all,ans_all,ans_all,
#              simple_complete    ,simple_complete,simple_complete,simple_complete,
#              heap_complete    ,heap_complete,heap_complete,heap_complete,
#              simple_single ,
#              simple_avg1   ,heap_avg1,
#              simple_avg2 ] #2454/3384 =>20:683/1173
# voting_li = [ans_all ,
#              simple_complete    ,
#              heap_complete    ,
#              simple_single ,
#              simple_avg1   ,heap_avg1,
#              simple_avg2, heap_avg2] #2454/3384 =>20:683/1173

                #4: #5:/2062/3467，6://3592，7:/2125/3286

for i,voting_li in enumerate(voting_lis):
    pd8 = pd.DataFrame(0,index=range(1095),columns=range(8))
    pd13 = pd.DataFrame(0,index=range(1095),columns=range(13))
    pd20 = pd.DataFrame(0,index=range(1095),columns=range(20))
    for method in voting_li : #20,13,8
        sort20 = sorted(method[0],key=len)
        sort13 = sorted(method[1],key=len)
        sort8 = sorted(method[2],key=len)
        for c,cluster in enumerate(sort8): #8群
            for doc in cluster: #各群裡面成員
                pd8.loc[doc,c] +=1
        for c,cluster in enumerate(sort13):
            for doc in cluster:
                pd13.loc[doc,c] +=1
        for c,cluster in enumerate(sort20): #20群
            for doc in cluster: #各群裡面成員
                pd20.loc[doc,c] +=1
    sort13 = sorted(clf13,key=len)
    for c,cluster in enumerate(sort13):
        for doc in cluster:
            pd13.loc[doc,c] += int(len(voting_li)/1.3)
    if i ==0:
        df20 = pd20
        print(len(voting_li))
    elif i ==1:
        df13 = pd13
        print(len(voting_li))
    elif i == 2:
        df8 = pd8
        print(len(voting_li))


# In[20]:


for df in [df8,df13,df20]:
    df['vote'] = 0
    for i in range(len(df)):
        df.loc[i,'vote'] = df.loc[i].idxmax()


# In[21]:


counts = 0
counts_minimize = 0
for pd_ in [df8,df13,df20]:
    cols = pd_.columns.tolist()[:-1]
    count = 0
    count_minimize = 0
    for col in cols:
        count += len(pd_[pd_[col]>int(len(voting_li)/2 )])
    pd_2 = pd_.drop(['vote'],axis=1)
    for i in range(len(pd_2)):
        count_minimize += pd_2.loc[i].isin([pd_2.loc[i].max()]).sum()
    
    print(count,count_minimize)
    counts+=count
    counts_minimize += count_minimize
print("ALL:",counts,'Minimize:',counts_minimize)


# In[22]:


for i in range(13):
    print(df13[df13['vote']==i].index.tolist())
print('======================================================================================')
for i in range(20):
    print(df20[df20['vote']==i].index.tolist())
print('======================================================================================')
for i in range(8):
    print(df8[df8['vote']==i].index.tolist())


# In[25]:


for df in [df20,df13,df8]:
    clus = df.columns.tolist()[:-1]
    f = open('output/'+str(len(clus))+'.txt','w')
    for clu_num in clus:
        for doc_id in df[df['vote']==clu_num].index.tolist(): #群集從小的開始寫入，最後一個cluster個數最多
            f.write(str(doc_id+1)+'\n')
#             f.write('\n')
        f.write('\n')
    f.close()


# In[24]:


df8

