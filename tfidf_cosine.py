#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\ardrakr\Downloads\archive (5)\india-news-headlines.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated()


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.duplicated().sum()


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[10]:


# Initialize the TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                             #min_df=8,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True) # Prevents divide-by-zero errors)


# In[11]:


# Calculate the TF-IDF scores for the documents
tfidf_matrix = tfidf.fit_transform(df['headline_text'])


# In[12]:


print(tfidf_matrix) 


# In[13]:


df_tfidf = pd.DataFrame(tfidf_matrix[0].T.todense(), index = tfidf.get_feature_names(), columns=["TF-IDF"])

df_tfidf = df_tfidf.sort_values('TF-IDF', ascending=False)


# In[14]:


df_tfidf


# In[15]:


df_tfidf2 = pd.DataFrame(tfidf_matrix[1].T.todense(), index = tfidf.get_feature_names(), columns=["TF-IDF"])

df_tfidf2 = df_tfidf2.sort_values('TF-IDF', ascending=False)


# In[16]:


df_tfidf2


# In[17]:


from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


doc2_tfidf=tfidf.transform(["pak attack"])
# calculate the cosine similarity between the documents
sim = cosine_similarity(tfidf_matrix, doc2_tfidf)
print(sim[0][0])


# In[19]:


text_content=df['headline_text']


# In[20]:


doc2_tfidf=tfidf.transform(["Status quo will not be disturbed "])
# calculate the cosine similarity between the documents
sim1 = cosine_similarity(tfidf_matrix, doc2_tfidf).flatten()


# In[24]:


print(sim1)


# In[21]:


# sort the headlines by cosine similarity and print the top results
related_headlines_indices = sim1.argsort()[:-10:-1]
print("Top related headlines:")
for i in related_headlines_indices:
    print(text_content[i])


# In[22]:


print(related_headlines_indices)


# In[23]:


doc4_tfidf=tfidf.transform(["Fissures in Hurriyat over Pak visit"])
# calculate the cosine similarity between the documents
sim2 = cosine_similarity(tfidf_matrix, doc4_tfidf)
print(sim2[1][0])


# In[29]:


# Get user query
query = input("Enter your query: ")
doc3_tfidf=tfidf.transform([query])
# calculate the cosine similarity between the documents
sim3 = cosine_similarity(tfidf_matrix, doc3_tfidf).flatten()


# In[30]:


print(sim3)


# In[31]:


# sort the headlines by cosine similarity and print the top results
related_headlines_indices = sim3.argsort()[:-5:-1]
print("Top related headlines:")
for i in related_headlines_indices:
    print(text_content[i])


# In[ ]:




