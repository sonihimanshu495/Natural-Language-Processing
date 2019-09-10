#!/usr/bin/env python
# coding: utf-8

# In[28]:


import nltk
import pandas as pd


# In[29]:


#nltk.download_shell()


# In[30]:


pwd


# # Data Collection And Visualization

# In[32]:


messages = [line.rstrip() for line in open(r'C:\Users\DELL\Desktop\NLP\Natural-Language-Processing\smsspamcollection\SMSSpamCollection')]


# In[33]:


print(len(messages))


# In[34]:


messages[50]


# In[35]:


for mess_no,message in enumerate(messages[0:10]):
    print(mess_no,message)
    print("/n")


# In[36]:


messages[0]


# In[37]:


import pandas as pd


# In[43]:


messages = pd.read_csv(r"C:\Users\DELL\Desktop\NLP\Natural-Language-Processing\smsspamcollection\SMSSpamCollection",sep = '\t',names = ['label','message']
)


# In[40]:


messages.head()


# In[45]:


messages.describe()


# In[46]:


messages.groupby('label').describe()


# In[50]:


messages['length'] = messages['message'].apply(len)


# In[51]:


messages.head()


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


messages['length'].plot.hist(bins= 50)


# In[56]:


messages['length'].describe()


# In[59]:


messages[messages['length'] == 910]['message'].iloc[0]


# In[61]:


messages.hist(column = 'length', by = 'label', bins = 100, figsize = (12,4))


# In[62]:


import string


# In[71]:


#string.punctuation


# In[72]:


from nltk.corpus import stopwords


# In[73]:


#stopwords.words('english')


# # Text Data Preprocessing - Tokenization
# 

# In[80]:


def text_process(mess):
    # REMOVE PUNCTUATION
    # REMOVE STOPWORDS
    # RETURN LIST OF CLEAN TEXT WORDS
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split()if word.lower() not in stopwords.words('english')]


# In[81]:


messages.head()


# In[85]:


#list(map(text_process,messages['message'].head(1)))
messages['message'].head().apply(text_process)


# # Vectorization

# In[86]:


from sklearn.feature_extraction.text import CountVectorizer


# In[87]:


trainer = CountVectorizer(analyzer = text_process )


# In[88]:


learner = trainer.fit(messages['message'])


# In[89]:


print(len(learner.vocabulary_))


# In[91]:


mess4 = messages['message'][3]


# In[92]:


print(mess4)


# In[93]:


learner4 = learner.transform([mess4])


# In[94]:


print(learner4)


# In[95]:


print(learner4.shape)


# In[96]:


new_messages = learner.transform(messages['message'])


# In[97]:


print('shape of sparse matrix: ',new_messages.shape)


# In[98]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[99]:


tfidf_transformer = TfidfTransformer().fit(new_messages)


# In[101]:


tfidf4 = tfidf_transformer.transform(learner4)


# In[102]:


print(tfidf4)


# In[106]:


messages_tfidf = tfidf_transformer.transform(new_messages)


# # Model Creation

# In[107]:


from sklearn.naive_bayes import MultinomialNB


# In[108]:


spam_detection_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[111]:


spam_detection_model.predict(tfidf4)[0]


# In[112]:


all_pred = spam_detection_model.predict(messages_tfidf)


# In[113]:


all_pred


# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:


msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size = 0.3) 


# In[117]:


msg_train.head()


# In[118]:


from sklearn.pipeline import Pipeline


# In[119]:


pipeline = Pipeline([
    ('trainer',CountVectorizer(analyzer = text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# In[120]:


pipeline.fit(msg_train,label_train)


# In[121]:


predictions = pipeline.predict(msg_test)


# In[122]:


from sklearn.metrics import classification_report


# In[123]:


print(classification_report(label_test,predictions))


# In[ ]:




