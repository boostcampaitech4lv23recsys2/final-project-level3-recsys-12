#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd


# In[92]:


item = pd.read_csv("../crawling/output/item.tsv", sep="\t")


# In[93]:


item.head(1)


# In[94]:


item.rating.fillna("0.0", inplace=True)


# In[95]:


item[item.rating == ("ņĀÉ 5.0ņĀ")]


# In[96]:


item[item.rating == ('м†Р 4.8м†')].index


# In[97]:


item.drop(item[item.rating == ("м†Р 4.8м†")].index, inplace=True)
item.drop(item[item.rating == ("ņĀÉ 5.0ņĀ")].index, inplace=True)


# In[98]:


item.rating = item.rating.astype(float)
item.review = item.review.astype(int)


# In[99]:


import numpy as np

bins = list(np.arange(0, 5.5, 0.5))
item["rating_binned"] = pd.cut(item["rating"], bins)


# In[100]:


item.rating_binned.value_counts()[:]


# In[101]:


item.rating_binned.value_counts()[:1].sum()


# In[102]:


item.rating_binned.value_counts()[1:].sum()


# In[103]:


import numpy as np

bins = list(np.arange(0, 5.5, 1))
item["rating_binned"] = pd.cut(item["rating"], bins)


# In[104]:


item.rating_binned.value_counts()[:]


# In[105]:


NEGATIVE_RATING = 4.0


# In[106]:


item = item[~(item.rating == 0)]


# In[107]:


item[(item.rating <= NEGATIVE_RATING)]


# In[108]:


item.shape


# In[147]:



import math
import numpy as np


def get_rating(rating, review):
    if review == 0:
        review = 1
    rating2neg_user = {1:1, 2:1, 3:2, 4:3}
    rating2pos_user = {1:4.5, 2:4.5, 3:4.5, 4:5}
    neg_socre, pos_score = 0, 0
    
    if rating >= 4:
        pos_score = rating2pos_user[4]
        neg_score = rating2neg_user[4]
    elif rating >= 3:
        pos_score = rating2pos_user[3]
        neg_score = rating2neg_user[3]
    elif rating >= 2:
        pos_score = rating2pos_user[2]
        neg_score = rating2neg_user[2]
    elif rating >= 1:
        pos_score = rating2pos_user[2]
        neg_score = rating2neg_user[2]
    else:
        pos_score = rating2pos_user[1]
        neg_score = rating2neg_user[1]
    
    A = np.array([[pos_score, neg_score], [1, 1]])
    B = np.array([rating, review])
    n_pos, n_neg = np.linalg.solve(A, B)

    total_reviews = review
    positve_reviews = n_pos
    review_score = positve_reviews / total_reviews
    rating = round(review_score - (review_score - 0.5) * math.pow(-math.log(total_reviews + 1), 10), 2)
    
    return rating * 100


# In[148]:


ratings = []
for rate, review in zip(item.rating, item.review):
    ratings.append(get_rating(rate, review))


# In[149]:


item["steam_rating"] = ratings


# In[150]:


item.sort_values("steam_rating", inplace=True, ascending=False)


# In[153]:


item.review.max()


# In[151]:


item


# In[152]:


import pickle

with open("popularity.pickle", "wb") as pkl:
    pickle.dump(list(item[:200].item), pkl)


# In[159]:


import pickle

with open("../gcp/popularity.pickle", "rb") as pkl:
    data = pickle.load(pkl)


# In[155]:


len(data)


# In[158]:


type(data[6])


# In[ ]:




