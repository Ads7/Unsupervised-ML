
# coding: utf-8

# In[41]:


import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
from sklearn.mixture import GaussianMixture
from Exam.PB2.soft_clustering_measure import v_measure


BASE_DIR = '/Users/aman/Dropbox/CS6220_Amandeep_Singh/Exam/PB2'
DATA_DIR = '{0}/sample_dataset1.txt'.format(BASE_DIR)
N_FEATURES = 1000
N_TOP_WORDS = 20
T = 20
K = 20


# In[6]:


# with open('/Users/aman/Dropbox/CS6220_Amandeep_Singh/Exam/PB2/whole_dataset.txt', 'r') as file:
#     data = file.readlines()
#     id = []
#     topic = []
#     text = []
#     for line in data:
#         info = line.split(',')
#         id.append(info[0])
#         topic.append(info[1][1:-1])
#         text.append(''.join(info[2:]).replace('"', ''))
#     d = {'id': id, 'topic': topic, 'text': text}
#     df = pd.DataFrame(data=d).to_csv(BASE_DIR + '/whole.csv')

# with open('/Users/aman/Dropbox/CS6220_Amandeep_Singh/Exam/PB2/whole_dataset.txt', 'r') as file:
#     data = file.readlines()
#     id = []
#     topic = []
#     text = []
#     for line in data:
#         info = line.split(',')
#         id.append(info[0])
#         topic.append(info[1][2:-1])
#         text.append(''.join(info[2:]).replace('"', ''))
#     d = {'id': id, 'topic': topic, 'text': text}
#     df = pd.DataFrame(data=d).to_csv(BASE_DIR + '/whole.csv')


# In[8]:


whole_data = pd.read_csv(BASE_DIR + '/whole.csv', names=['id', 'text', 'topic'], skiprows=1)
sample_data = pd.read_csv(BASE_DIR + '/new.csv', names=['id', 'text', 'topic'], skiprows=1)


# In[9]:


tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,max_features=N_FEATURES,stop_words='english')
tf = tf_vectorizer.fit_transform(whole_data.text.tolist()).todense()


# In[17]:


lda = LatentDirichletAllocation(n_components=T, learning_method='online')
lda_transform = lda.fit_transform(tf)


# In[20]:


gm = GaussianMixture(n_components=K)


# In[21]:


gm.fit(lda_transform[sample_data.id.tolist(),:])


# In[43]:


# predictions = gm.predict(lda_transform[sample_data.id.tolist(),:])
label_true = [np.argmax(lda.transform(i)[0]) for i in tf[sample_data.id.tolist()]] 
labels_pred_prob = gm.predict_proba(lda_transform[sample_data.id.tolist(), :])
print(v_measure(labels_pred_prob,label_true,20,20))


# In[45]:


top_topics = np.argpartition((np.sum(gm.predict_proba(lda_transform[sample_data.id.tolist(), :]),axis=0)),-5)[-5:]


# In[53]:


n_top_words = 20
feature_names = tf_vectorizer.get_feature_names()
# def print_top_words(lda_transform, feature_names, n_top_words):
for topic_idx, topic in enumerate(lda.components_[top_topics]):
    message = "Topic #%d: " % topic_idx
    index = topic.argsort()[:-n_top_words - 1:-1]
    message += " ".join([feature_names[i] + " " + str(topic[i] / topic.sum()) for i in index])
    print(message)

