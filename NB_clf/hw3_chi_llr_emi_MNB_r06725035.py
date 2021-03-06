#!/usr/bin/env python
# coding: utf-8

# ### R06725035 陳廷易
# * feature selection: chi-square, log likelihood ratios, expected mutual information
# * naive bayes classification
# * voting

# In[1]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import os,sys
# import re
import pickle
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
import random
import math
from collections import Counter
import functools


# In[2]:


with open('data/stop_words.txt') as f:
    stop_words_list = f.read().splitlines() #stop_list1
stop_list2 = pickle.load(open('data/stop_list2.pkl','rb'))
ps = PorterStemmer() # Stemming
stop_words = set(stopwords.words('english')) #Stopword
short = ['.', ',', '"', "\'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "'at",
         "_","`","\'\'","--","``",".,","//",":","___",'_the','-',"'em",".com",
                   '\'s','\'m','\'re','\'ll','\'d','n\'t','shan\'t',"...","\'ve",'u']
stop_words_list.extend(short)
stop_words_list.extend(stop_list2)
stop_words.update(stop_words_list)


# In[3]:


def preprocess(texts):
    tokens = [i for i in word_tokenize(texts.lower()) if i not in stop_words]  # Tokenization.# Lowercasing
    token_result = ''
    token_result_ = ''
    for i,token in enumerate(tokens): #list2str
        token_result += ps.stem(token) + ' '
    token_result = ''.join([i for i in token_result if not i.isdigit()])
    token_result = [i for i in word_tokenize(token_result) if i not in stop_words]
    for i,token in enumerate(token_result):
        token_result_ += token + ' '
    return token_result_


# ## Feature Selection
# * tf-idf
# * chi-square
# * likelihood
# * PMI
# * EMI
# 
# => build dictionary in 500 words

# In[4]:


dict_df = pd.read_csv('data/dictionary.txt',header=None,index_col=None,sep=' ')
terms = dict_df[1].tolist() #all terms


# In[5]:


with open('data/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_dict[trainid[0]] = trainid[1:]
print(train_dict) #class:doc_id
train_dict = pickle.load(open('data/train_dict.pkl','rb'))


# In[6]:


in_dir = 'data/IRTM/'
train_dict_ = {}
class_token = []
class_token_dict = {}
for c,d in train_dict.items():
    for doc in d:
        f = open('data/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocess(texts)
        tokens_all = tokens_all.split(' ')
        tokens_all = list(set(filter(None,tokens_all)))
        class_token.append(tokens_all)
    class_token_dict[c]=class_token
    class_token=[]
# len(class_token_dict['1'])


# In[7]:


dict_df.drop(0,axis=1,inplace=True)
dict_df.columns = ['term','score']
dict_df.index = dict_df['term']
dict_df.drop('term',axis=1,inplace=True)
print(dict_df)


# In[8]:


dict_df['score'] = 0
dict_df['score_chi'] = 0
dict_df['score_emi'] = 0
# c=1
for term in tqdm(terms): #each term
    scores = []
    scores_chi = []
    scores_emi = []
    c=1
    for _ in range(len(class_token_dict)): # each class
        n11=e11=m11=0
        n10=e10=m10=0
        n01=e01=m01=0
        n00=e00=m00=0
        for k,v in class_token_dict.items():
#             print(k,c)
            if k == str(c): #ontopic
                for r in v:
                    if term in r:
                        n11+=1
                    else:
                        n10+=1
#                 c+=1
            else: #off topic
                for r in v:
                    if term in r:
                        n01+=1
                    else:
                        n00+=1
#                 c+=1
        c+=1
        n11+=1e-8
        n10+=1e-8
        n01+=1e-8
        n00+=1e-8
        N = n11+n10+n01+n00
        e11 = N * (n11+n01)/N * (n11+n10)/N #chi-squre
        e10 = N * (n11+n10)/N * (n10+n00)/N
        e01 = N * (n11+n01)/N * (n01+n00)/N
        e00 = N * (n01+n00)/N * (n10+n00)/N
        score_chi = ((n11-e11)**2)/e11 + ((n10-e10)**2)/e10 + ((n01-e01)**2)/e01 + ((n00-e00)**2)/e00
        scores_chi.append(score_chi)
        
        n11 = n11 - 1e-8 + 1e-6
        n10 = n10 - 1e-8 + 1e-6
        n01 = n01 - 1e-8 + 1e-6
        n00 = n00 - 1e-8 + 1e-6
        N = n11+n10+n01+n00
        m11 = (n11/N) * math.log(((n11/N)/((n11+n01)/N * (n11+n10)/N)),2) #EMI
        m10 = n10/N * math.log((n10/N)/((n11+n10)/N * (n10+n00)/N),2)
        m01 = n01/N * math.log((n01/N)/((n11+n01)/N * (n01+n00)/N),2)
        m00 = n00/N * math.log((n00/N)/((n01+n00)/N * (n10+n00)/N),2)
        score_emi = m11 + m10 + m01 + m00
        scores_emi.append(score_emi)
        
#         print(n11,n10,n01,n00)
        n11-=1e-6
        m10-=1e-6
        n01-=1e-6
        n00-=1e-6
        N = n11+n10+n01+n00
        score = (((n11+n01)/N) ** n11) * ((1 - ((n11+n01)/N)) ** n10) * (((n11+n01)/N) ** n01) * ((1 - ((n11+n01)/N)) ** n00)
        score /= ((n11/(n11+n10)) ** n11) * ((1 - (n11/(n11+n10))) ** n10) * ((n01/(n01+n00)) ** n01) * ((1 - (n01/(n01+n00))) ** n00)
        score = -2 * math.log(score, 10) #LLR
        scores.append(score)
        
        
#         c+=1
    dict_df.loc[term,'score'] = np.mean(scores)
    dict_df.loc[term,'score_chi'] = np.mean(scores_chi)
    dict_df.loc[term,'score_emi'] = np.mean(scores_emi)
print(dict_df)    


# In[9]:


dict_df2 = pd.read_csv('data/dictionary.txt',header=None,index_col=None,sep=' ')
dict_df2.columns = ['id','term','freq']
dict_df2['sum'] = 0.0
print(dict_df2)


# In[10]:


# import os
tf_list = next(os.walk('data/tf-idf/'))[2]
# df_list = [dict_df2]
for tf in tf_list:
#     print(tf)
    df2 = pd.read_csv('data/tf-idf/'+tf,header=None,index_col=None,sep=' ',skiprows=[0])
    df2.columns = ['id','tfidf']
    df3 = pd.merge(dict_df2,df2,on='id',how='outer')
    df3.fillna(0,inplace=True)
    dict_df2['sum']+=df3['tfidf']
dict_df2['avg_tfidf'] = dict_df2['sum']/dict_df2['freq']
dict_df2 = dict_df2.drop(['freq','sum'],axis=1)
print(dict_df2)
#     break
#     df_list.append(df2)
# df3 = pd.concat(df_list).groupby(level=0).sum()
# df3 = pd.concat([df3,df2]).groupby(level=0).sum()
# df3 = pd.merge(dict_df2,df2,on='id',how='outer')
# df3 = pd.merge(df3,df2,on='id',how='outer')?
# df3[df3.id == 2]


# In[11]:


dict_df['term'] = dict_df.index
dict_df3 = pd.merge(dict_df,dict_df2,on='term',how='outer')
print(dict_df3)


# In[12]:


cols = list(dict_df3)
cols[4], cols[3], cols[5], cols[1], cols[0], cols[2] = cols[0], cols[1] , cols[2] , cols[3], cols[4], cols[5]
dict_df3 = dict_df3.ix[:,cols]
print(dict_df3)


# In[13]:


dict_df3.columns = ['id','term','avg_tfidf','score_chi','score_llr','score_emi']
# dict_df3.to_csv('output/feature_selection_df_rev.csv')


# ### select top 500
# * 取各col的mean+1.45*std
# * 再去做投票，超過兩票的流下來看剩下哪幾個

# In[14]:


dict_df3 = pd.read_csv('output/feature_selection_df_rev.csv',index_col=None)
threshold_tfidf = np.mean(dict_df3['avg_tfidf'])+2.5*np.std(dict_df3['avg_tfidf']) #1.45=>502 數字大嚴格
threshold_chi = np.mean(dict_df3['score_chi'])+2.5*np.std(dict_df3['score_chi']) #1=>350
threshold_llr = np.mean(dict_df3['score_llr'])+2.5*np.std(dict_df3['score_llr']) #1.75=>543
threshold_emi = np.mean(dict_df3['score_emi'])+2.5*np.std(dict_df3['score_emi']) #1.75=>543

print('avg_tfidf',threshold_tfidf)
# dict_df3[dict_df3.score_llr>0.1]


# In[15]:


df1 = dict_df3[dict_df3['avg_tfidf']>threshold_tfidf]
df2 = dict_df3[dict_df3['score_chi']>threshold_chi]
df3 = dict_df3[dict_df3['score_llr']>threshold_llr]
df4 = dict_df3[dict_df3['score_emi']>threshold_emi]
df_vote = dict_df3
df_vote['vote']=0
print(df_vote)


# In[16]:


df_vote.loc[df1.id-1,'vote'] += 1
df_vote.loc[df2.id-1,'vote'] += 1
df_vote.loc[df3.id-1,'vote'] += 1
df_vote.loc[df4.id-1,'vote'] += 1
# df_vote


# In[17]:


df_vote_ = df_vote[df_vote.vote>2] #(1,2)=>375 #(1,1)=>422 #(1.6,2)=>482 #(2,2)=>330 #(1,3)=>100
df_vote_ = df_vote_.filter(['id','term','vote'])
print(df_vote_)


# In[18]:


# df_vote_.to_csv('output/500terms_df_rev5.csv')


# ## Classifier
# * 7-fold
# * MNB
# * BNB
# * self-train
# * ens voting (BNB lower weight)

# In[19]:


df_vote = pd.read_csv('output/500terms_df_rev5.csv',index_col=False)
terms_li = list(set(df_vote.term.tolist()))

train_X = []
train_Y = []
print(len(terms_li))


# In[20]:


with open('data/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}

for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_dict[trainid[0]] = trainid[1:]
# train_dict #class:doc_id
train_dict = pickle.load(open('data/train_dict.pkl','rb'))
in_dir = 'data/IRTM/'
train_dict_ = {}
class_token = []
class_token_dict = {}
train_X = []
train_Y= []
train_ids = []
for c,d in tqdm(train_dict.items()):
    for doc in d:
        train_ids.append(doc)
        trainX = np.array([0]*len(terms_li))
        f = open('data/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocess(texts)
        tokens_all = tokens_all.split(' ')
#         tokens_all = list(filter(None,tokens_all))
        tokens_all = dict(Counter(tokens_all))
        for key,value in tokens_all.items():
            if key in terms_li:
                trainX[terms_li.index(key)] = int(value)
#         trainX = np.array(trainX)
        
#         for token in tokens_all:
#             if token in terms_li:
#                 ind = terms_li.index(token)
#                 trainX[ind]+=1
        train_X.append(trainX)
        train_Y.append(int(c))
        
train_X = np.array(train_X)
train_Y = np.array(train_Y)

#         tokens_all = list(set(filter(None,tokens_all)))
#         class_token.append(tokens_all)
#     class_token_dict[c]=class_token
#     class_token=[]
# len(class_token_dict['1'])
# print(train_X.shape , train_Y.shape)


# In[21]:


#建立term index matrix
tokens_all_class=[]
term_tf_mat=[]
for c,d in tqdm(train_dict.items()):
    for doc in d:
        f = open('data/IRTM/'+doc+'.txt')
        texts = f.read()
        f.close()
        tokens_all = preprocess(texts)
        tokens_all = tokens_all.split(' ')
        tokens_all = list(filter(None,tokens_all))
        tokens_all_class.extend(tokens_all)
    tokens_all = dict(Counter(tokens_all_class))
    term_tf_mat.append(tokens_all)


# In[22]:


def train_MNB(train_set=train_dict,term_list=terms_li,term_tf_mat=term_tf_mat):
    prior = np.zeros(len(train_set))
    cond_prob = np.zeros((len(train_set), len(term_list)))
    
    for i,docs in train_set.items(): #13 classes 1~13
        prior[int(i)-1] = len(docs)/len(train_ids) #那個類別的文章有幾個/總共的文章數目 0~12
        token_count=0
        class_tf = np.zeros(len(term_list))
        for idx,term in enumerate(term_list):
            try:
                class_tf[idx] = term_tf_mat[int(i)-1][term]  #term在class的出現次數
            except:
                token_count+=1

        class_tf = class_tf + np.ones(len(term_list)) #smoothing (可改)
        class_tf = class_tf/(sum(class_tf) +token_count) #該class總共的token數(可改)
        cond_prob[int(i)-1] = class_tf #0~12
    return prior, cond_prob


# In[23]:


prior,cond_prob = train_MNB()
print(prior,cond_prob)


# In[24]:


def predict_MNB(test_id,prob=False,prior=prior,cond_prob=cond_prob,term_list=terms_li):
    f = open('data/IRTM/'+str(test_id)+'.txt')
    texts = f.read()
    f.close()
    tokens_all = preprocess(texts)
    tokens_all = tokens_all.split(' ')
    tokens_all = list(filter(None,tokens_all))
    
    class_scores = []
#     score = 0
    for i in range(13):
        score=0
#         print(prior[i])
        score += math.log(prior[i],10)
        for token in tokens_all:
            if token in term_list:
                score += math.log(cond_prob[i][term_list.index(token)])
        class_scores.append(score)
    if prob:
        return np.array(class_scores)
    else:
        return(np.argmax(class_scores)+1)


# In[ ]:





# for testing class function only

# In[25]:


def MNB(input_X,input_Y=None,prior_log_class=None,log_prob_feature=None,train=True,prob=False,smooth=1.0):
    if train:
        sample_num = input_X.shape[0]
        match_data = [[x for x, t in zip(input_X, input_Y) if t == c] for c in np.unique(input_Y)]
        prior_log_class = [np.log(len(i) / sample_num) for i in match_data]
        counts = np.array([np.array(i).sum(axis=0) for i in match_data]) + smooth
        log_prob_feature = np.log(counts / counts.sum(axis=1)[np.newaxis].T)
        return prior_log_class,log_prob_feature
    else:
        probability = [(log_prob_feature * x).sum(axis=1) + prior_log_class for x in input_X]
        if prob:
            return probability
        else:
            ans = np.argmax(probability,axis=1)
            return ans


# In[26]:


class BernoulliNB(object):
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize
    def _binarize_X(self, X):
        return np.where(X > self.binarize, 1, 0) if self.binarize != None else X
    def fit(self, X, y):
        X = self._binarize_X(X)
        count_sample = X.shape[0]
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha
        smoothing = 2 * self.alpha
        n_doc = np.array([len(i) + smoothing for i in separated])
        self.feature_prob_ = count / n_doc[np.newaxis].T
        return self

    def predict_log_proba(self, X):
        X = self._binarize_X(X)
        return [(np.log(self.feature_prob_) * x +                  np.log(1 - self.feature_prob_) * np.abs(x - 1)
                ).sum(axis=1) + self.class_log_prior_ for x in X]

    def predict(self, X):
        X = self._binarize_X(X)
        return np.argmax(self.predict_log_proba(X), axis=1)


# ### Prediction

# In[27]:


df_vote = pd.read_csv('output/500terms_df_rev5.csv',index_col=False)
terms_li = list(set(df_vote.term.tolist()))
print(len(terms_li))


# In[28]:


with open('data/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
test_id = []
train_ids=[]
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_ids.extend(trainid[1:])
for i in range(1095):
    if str(i+1) not in train_ids:
        test_id.append(i+1)
ans=[]
for doc in tqdm(test_id):
    ans.append(predict_MNB(doc))
print(ans)
df_ans = pd.DataFrame(list(zip(test_id,ans)),columns=['id','Value'])
# df_ans.to_csv('output/MNB.csv',index=False)
# df_ans


# combine all prediction df

# In[40]:


# import os
in_dir = './data/'
prefixed = [filename for filename in os.listdir(in_dir) if filename.startswith("MNB")]
df_from_each_file = [pd.read_csv(in_dir+f) for f in prefixed]
# prefixed


# In[41]:


merged_df = functools.reduce(lambda left,right: pd.merge(left,right,on='id'), df_from_each_file)
merged_df.columns = ['id',0,1,2,3,4,5,6,7,8]
print(merged_df)


# In[42]:


df01 = pd.DataFrame(merged_df.mode(axis=1)[0])
df02 = pd.DataFrame(merged_df['id'])
df_ans = pd.concat([df02,df01],axis=1)
df_ans = df_ans.astype('int')
df_ans.columns = ['id','Value']
df_ans.to_csv('output/voting_rev2.csv',index=False)
print(df_ans)


# In[ ]:





# In[ ]:





# do only in first time only ro produce new train_dict

# In[32]:


# df1 = merged_df[(merged_df[0] == merged_df[1])&(merged_df[2]==merged_df[3])&(merged_df[1]==merged_df[2])]
# df1.reset_index(inplace=True,drop=True)
# df1['class'] = df1[0]
# df1 = df1.filter(['id','class'])
# df1


# In[33]:


# df1[df1['class']=='1']


# In[34]:


# with open('data/training.txt','r') as f:
#     train_id = f.read().splitlines()
# train_dict = {}
# for trainid in train_id:
#     trainid = trainid.split(' ')
#     trainid = list(filter(None, trainid))
#     train_dict[trainid[0]] = trainid[1:]
# train_dict #class:doc_id


# In[35]:


# for i in range(13):
#     df1 = df1.astype(str)
#     li = df1[df1['class'] == str(i+1)]['id'].tolist()
#     train_dict[str(i+1)].extend(li)
# train_dict


# In[36]:


# pickle.dump(obj=train_dict,file=open('data/train_dict.pkl','wb'))


# In[ ]:





# In[ ]:





# In[37]:


with open('data/training.txt','r') as f:
    train_id = f.read().splitlines()
train_dict = {}
test_id = []
train_ids=[]
for trainid in train_id:
    trainid = trainid.split(' ')
    trainid = list(filter(None, trainid))
    train_ids.extend(trainid[1:])
for i in range(1095):
    if str(i+1) not in train_ids:
        test_id.append(i+1)
#     train_dict[trainid[0]] = trainid[1:]
# train_dict #class:doc_id
in_dir = 'data/IRTM/'
train_dict_ = {}
class_token = []
class_token_dict = {}
test_X = []
# train_Y= []
# for c,d in tqdm(train_dict.items()):
for doc in tqdm(test_id):
    testX = np.array([0]*len(terms_li))
    f = open('data/IRTM/'+str(doc)+'.txt')
    texts = f.read()
    f.close()
    tokens_all = preprocess(texts)
    tokens_all = tokens_all.split(' ')
#         tokens_all = list(filter(None,tokens_all))
    tokens_all = dict(Counter(tokens_all))
    for key,value in tokens_all.items():
        if key in terms_li:
            testX[terms_li.index(key)] = int(value)

    test_X.append(testX)
        
test_X = np.array(test_X)
# print(test_X.shape)


# In[38]:


# print(train_X.shape, train_Y.shape , test_X.shape)


# In[ ]:





# In[39]:


# df_ans = pd.DataFrame(list(zip(test_id,ans2)),columns=['id','Value'])
# df_ans.to_csv('output/MNB02.csv',index=False)
# df_ans

