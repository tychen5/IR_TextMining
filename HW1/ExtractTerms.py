
# coding: utf-8

# ### R06725035 陳廷易
# * Tokenization.
# * Lowercasing everything.
# * Stemming using Porter’s algorithm.
# * Stopword removal.
# * Save the result as a txt file. 
# 

# In[31]:


# import keras
# from keras.preprocessing.text import Tokenizer
# import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
#nltk.download('all')


# ### read data

# In[55]:


file = open('data/28.txt','r')
texts = file.read()
print(texts)


# ### main preprocessing

# In[64]:


ps = PorterStemmer() # Stemming
stop_words = set(stopwords.words('english')) #Stopword
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',
                   '\'s','\'m','\'re','\'ll','\'d','n\'t','shan\'t']) # remove it if you need punctuation

tokens = [i for i in word_tokenize(texts.lower()) if i not in stop_words]  # Tokenization.# Lowercasing
token_result = ''
for i,token in enumerate(tokens):
    if i != len(tokens)-1: # 最後不要空白
        token_result += ps.stem(token) + ' '
    else:
        token_result += ps.stem(token)

# tokens = nltk.word_tokenize(texts.lower())
# ps.stem(token_result)


# ### Output

# In[67]:


# output=""
# for token in tokens:
#     output+=token+' '
# print(output)
file = open('result/output.txt','w')
file.write(token_result) #Save the result 
file.close()
print(token_result)


# In[14]:


# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(texts)
# print(tokenizer.sequences_to_texts())

