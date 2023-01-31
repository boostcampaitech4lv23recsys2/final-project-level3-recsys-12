#!/usr/bin/env python
# coding: utf-8

# In[96]:


import pandas as pd


# In[97]:


item = pd.read_csv("/opt/ml/input/data/item.csv")


# In[98]:


item.title.apply(lambda x: len(x)).mean()


# In[99]:


# TODO 모든 title을 전처리합니다. -> tokenizer로 전처리한 text 중에 길이가 5이하인 애들이 있습니다.
# 전처리 전의 title이지만 평균 길이가 23이기에, 전처리 이후의 text 길이가 5이하인 애들은 제대로 클러스터링된 아이템들이 아닙니다. -> 클러스터링할 때 이 부분 고려합니다.
# 전처리 후에 전처리 결과를 item df에 merge합니다.
# TODO 전처리한 title의 similarity를 계산합니다. -> 전처리 title을 기준으로 item df를 sort합니다 -> 게슈탈트 패턴 매칭 알고리즘을 사용해서 계산합니다. -> 이는 재사용을 위해 pickle 객체로 저장합니다.
# TODO item df를 전처리한 title 기준으로 sort 후에, 마찬가지로 전처리한 title 기준으로 정렬된 pickle 객체를 불러와서 merge합니다.
# TODO 추천에 활용하지 않을 category 목록들을 필터링합니다.
# TODO 필터링 이후, similarity에 대한 BFS 알고리즘을 적용합니다 -> 이전에 구한 similarity가 전체 item을 기준으로 구해졌기 때문에, 예외 처리를 해줘야 합니다.
# TODO BFS 알고리즘을 통해 찾아낸 clustering group을 만들어주고, "ETC"나 "etc"로 group된 item들은 -1로 변환합니다.
# TODO house_interaction에 클러스터 결과를 merge해줍니다. 이때, 클러스터 그룹이 -1이거나 house_interaction에만 포함된 item들은 cluster_id.max() + 1을 더해줍니다.
# TODO model을 학습해서 학습한 결과를 backend에 반영합니다.
# TODO cluster-major-item list를 생성합니다.


# In[100]:


item["original_title"] = item.title


# In[101]:


item.rename(columns={"title":"preprocessed_title"}, inplace=True)


# In[102]:


def df_lower(df:pd.DataFrame):
    df.preprocessed_title = df.preprocessed_title.apply(lambda x:str.lower(x))
    return df

# TODO 전처리: [단종], [품목], (당일출고) 등등 -> "" -> 재정렬
def get_del_words(df:pd.DataFrame):
    del_words = set()
    from tqdm import tqdm
    for a in tqdm(df.preprocessed_title.unique()):
        tmp = a.strip("\t\n ")
        if tmp.startswith("[") and "]" in tmp:
            tmp = tmp.split("]")[0][1:]
            del_words.add("[" + tmp + "]")
        elif tmp.startswith("(") and ")" in tmp:
            tmp = tmp.split(")")[0][1:]
            del_words.add("(" + tmp + ")")
    return del_words

def df_strip(df:pd.DataFrame, del_words:str = "\t\n #&"):
    df.preprocessed_title = df.preprocessed_title.apply(lambda x:x.strip(del_words))
    return df

# TODO 전처리: 양끝에 \t, \n, " ", ﻿ -> ""
def df_strip2(df:pd.DataFrame):
    df.preprocessed_title = df.preprocessed_title.apply(lambda x:x.strip("\t\n ﻿"))
    return df

def df_del_word(df:pd.DataFrame, del_words:list):
    def f(x, word):
        if x.startswith(word):
            return x[len(word):].strip(" ")
        else:
            return x
    from tqdm import tqdm
    for word in tqdm(del_words):
        df.preprocessed_title = df.preprocessed_title.apply(lambda x:f(x, word))
    return df


# In[103]:


item = df_lower(item)


# In[104]:


del_list = get_del_words(item)
item = df_del_word(item, del_list)


# In[105]:


item = df_strip(item)


# In[106]:


item = df_strip2(item)


# In[107]:


item.preprocessed_title = item.preprocessed_title.str.replace(pat=r'[^\w]',repl=r' ',regex=True)


# In[108]:


item.preprocessed_title.fillna("ETC", inplace=True)


# In[145]:


from tensorflow.keras.preprocessing.text import text_to_word_sequence
from konlpy.tag import Okt
from tqdm import tqdm
import re

reg = re.compile(r'[a-zA-Z]')

def tokenize(x:str):
    okt = Okt() # 형태소 분석기 객체 생성
    noun_list = []
    x = x.split()
    for s in x:
        if reg.match(s):
            noun_list.extend(text_to_word_sequence(s))
        elif s.isdigit():
            noun_list.append(s)
        else:
            noun_list.extend(okt.nouns(okt.normalize(s)))
    return " ".join(noun_list)


# In[146]:


tokenize("Nintendo Switch 90 포트3나이트 스페셜 세트")


# In[147]:


tt = item[:]


# In[156]:


from tqdm import tqdm, tqdm_pandas
tqdm.pandas()

item.preprocessed_title = item.preprocessed_title.progress_apply(lambda x:tokenize(x))


# In[157]:


item.sort_values("item", inplace=True)


# In[158]:


from time import time

now = str(round(time()))[5:]

item.to_csv(f"title_preprocessed_item_{now}.csv", index=False)

