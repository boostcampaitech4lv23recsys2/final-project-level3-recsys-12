#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
from tqdm import tqdm

# In[57]:


# TODO house_interaction에 클러스터 결과를 merge해줍니다. 이때, 클러스터 그룹이 -1이거나 house_interaction에만 포함된 item들은 cluster_id.max() + 1을 더해줍니다.
# TODO model을 학습해서 학습한 결과를 backend에 반영합니다.
# TODO cluster-major-item list를 생성합니다.


# In[58]:


item = pd.read_csv("/opt/ml/input/final/EDA/clustered_item_32715.csv")
train = pd.read_csv("/opt/ml/input/data/hi_interaction.csv")


# In[59]:


item.head(1)


# In[60]:


train.head(1)


# In[61]:


item.cls_id.nunique()


# In[62]:


item.item.nunique()


# In[63]:


len(train)


# In[64]:


# item에 포함되지 않은 hi_interaction item을 제거하기
train = train[train.item.isin(item.item.unique())]


# In[65]:


len(train)


# In[66]:


item.cls_id.max()


# In[12]:


len(item[item.cls_id == -1])


# In[13]:


dummy = list()
c = item.cls_id.max()

for i in range(len(item[item.cls_id == -1])):
    dummy.append(i + c + 1)


# In[14]:


item.loc[item.cls_id == -1, "cls_id"] = dummy


# In[15]:


item.head(3)


# In[ ]:


item.cls_id.nunique()


# In[16]:


item2cls = {item_id: cls_id for item_id, cls_id in zip(item.item, item.cls_id)}


# In[17]:


train["cluster_id"] = train.item.map(item2cls)


# In[18]:


train


# In[19]:


train = train[["house", "cluster_id", "item"]]


# In[20]:


train = train.sort_values(by=["house", "cluster_id", "item"])


# In[21]:


train


# In[22]:


train.to_csv("clustered_train.csv", index=False)


# In[23]:


predict_price_list = []

import re

p = re.compile(".+(?=사)")
for i in item.predict_price.index:
    m = p.findall(str(item.predict_price.iloc[i]))
    if len(m) == 0:
        m.append("예상가정보없음")
    if m[0][3:] == "정보없음":
        predict_price_list.append("정보없음")
    else:
        predict_price_list.append(int("".join((m[0][3:]).split(","))))


# In[24]:


item.predict_price = predict_price_list


# In[25]:


item.loc[(item.predict_price != "정보없음"), "price"] = item[
    (item.predict_price != "정보없음")
].predict_price


# In[27]:


import re


def f(x):
    pattern = re.compile(r"별점\s+(\d+\.\d+)점")
    try:
        match = pattern.search(x)
        if match:
            score = float(match.group(1))
        else:
            score = None
        return score
    except:
        return 0.0


# In[28]:


item.rating = item.rating.apply(lambda x: f(x))


# In[29]:


item.review = item.review.fillna("리뷰 쓰기첫 리뷰 두 배 적립")
review_list = []

import re

p = re.compile("[(][0-9]*[)]")
for idx, review in enumerate(item.review):
    m = p.findall(review)
    if len(m) == 0:
        m.append("(0)")
    review_list.append(m[0][1:-1])

item.review = review_list


# In[30]:


item.review = item.review.astype(int)


# In[31]:


item.rating = item.rating.astype(float)


# In[32]:


item.head(5)


# In[33]:


item = item.sort_values(["cls_id", "review", "rating"], ascending=[True, False, False])


# In[35]:


item.groupby("cls_id")["item"].apply(lambda x: len(list(x)))


# <!-- 1. 클러스터링 적당히 엄격히하고 (현 수준) -> 생성된 테이블로 백, 프론트, DB 작업하고 -> 검증만 다중선형회귀로
# 1. AE 학습할 때 feature로 넣는다? -->

# In[ ]:


# In[46]:


cls_major_item_id = (
    item.rename(columns={"item": "major_item"})
    .groupby("cls_id")["major_item"]
    .apply(lambda x: list(x)[0])
    .reset_index()
)
cls_major_item_id_list = (
    item.rename(columns={"item": "item_list"})
    .groupby("cls_id")["item_list"]
    .apply(lambda x: "|".join(list(map(str, list(x)))))
    .reset_index()
)


# In[47]:


cls_major_item_id = cls_major_item_id.merge(cls_major_item_id_list, on="cls_id")


# In[48]:


cls_major_item_id


# In[49]:


cls_major_item_id.to_csv("cluster_major_item.csv", index=True)


# In[ ]:
