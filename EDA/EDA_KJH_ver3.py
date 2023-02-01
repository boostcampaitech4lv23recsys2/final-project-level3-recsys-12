#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
from tqdm import tqdm
from collections import Counter


# In[47]:


# TODO 추천에 활용하지 않을 category 목록들을 필터링합니다.
# TODO 필터링 이후, similarity에 대한 BFS 알고리즘을 적용합니다 -> 이전에 구한 similarity가 전체 item을 기준으로 구해졌기 때문에, 예외 처리를 해줘야 합니다.
# TODO BFS 알고리즘을 통해 찾아낸 clustering group을 만들어주고, "ETC"나 "etc"로 group된 item들은 -1로 변환합니다.
# TODO house_interaction에 클러스터 결과를 merge해줍니다. 이때, 클러스터 그룹이 -1이거나 house_interaction에만 포함된 item들은 cluster_id.max() + 1을 더해줍니다.
# TODO model을 학습해서 학습한 결과를 backend에 반영합니다.
# TODO cluster-major-item list를 생성합니다.


# In[48]:


item = pd.read_csv("similarity_item_99243.csv")


# In[49]:


item.head(1)


# In[50]:


del_list = [
"공구",
"욕실용품",
"유아·아동",
"청소용품",
"캠핑·레저",
"반려동물",
"생필품",
"렌탈",
"생활용품",
"식품",
"인테리어시공",
"칼",
"선풍기",
"컵·잔·텀블러",
"조리도구·도마",
"홈갤러리",
"전기요·온수매트",
# "에어컨",
"커피·티용품",
"조명부속품",
"센서등",
"복합기·프린터·스캐너",
"소파·생활커버",
"이미용가전",
"주방잡화",
"그릇·홈세트",
"후크·수납걸이",
"데스크·디자인문구",
"냄비·프라이팬·솥",
"베이킹용품",
# "세탁기·건조기",
"거울",
"매트리스·토퍼",
"파티·이벤트용품",
"해충퇴치기",
"다이어리·플래너",
"수세미걸이·세제통",
"다리미·보풀제거기",
"멀티탭·공유기정리함",
"멀티탭·공유기정리함",
"다리미·보풀제거기",
"디자인문구",
"보드게임",
"리빙박스·바구니",
# "가습기",
"전기히터·온풍기",
"전기·멀티포트"
]


# In[51]:


item.category.fillna("|", inplace=True)


# In[52]:


import pickle
with open("similarity_99243.pickle", "rb") as pkl:
    data = pickle.load(pkl)

item["similarity_list"] = data


# In[53]:


item_ = item[(~item.category.str.contains("|".join(del_list))) & ~(item.category=="|")]


# In[54]:


len(item_)


# In[55]:


(item_.iloc[6].similarity_list)


# In[56]:


item_.sort_values("item", inplace=True)


# In[57]:


list(item_[item_.item == 356808].similarity_list)


# In[58]:


CLUSTER_ID = 0
NON_CLUSTER_ID = -1

lim = 5
threshold = 0.8

# TODO threshold 이상의 similarity 가지는 아이템을 저장하는 list 칼럼 생성
def get_item(x, threshold):
    tmp = []
    for a in x:
        if a[3] >= threshold:
            tmp.append(a[2])
    return tmp

item_["similar_item_list"] = item_.similarity_list.apply(lambda x:get_item(x, threshold))


# In[59]:


item_[item_.item == 356808]


# In[60]:


graph = {item_id:sim_items for item_id, sim_items in zip(item_.item, item_.similar_item_list)}
visited = {item_id:NON_CLUSTER_ID for item_id in item_.item}


# In[61]:


# 양방향 그래프로 만들기
for item_id, sim_items in graph.items():
    for item_id_ in sim_items:
        try:
            graph[item_id_].append(item_id)
        except:
            pass


# In[62]:


len(graph)


# In[63]:


len(visited)


# In[64]:


cnt = 0
for g in graph.values():
    if len(g) > 1:
        cnt += 1
cnt


# In[65]:


def BFS(item_id:int, cls_id):
    from collections import deque
    q = deque([(item_id, 0)])
    visited[item_id] = cls_id
    while(q):
        node, hop = q.popleft()
        if hop > lim:
            return
        try:
            for id in graph[node]:
                if visited[id] == -1: # 아직 클러스터링되지 않은 아이템인 경우
                    visited[id] = cls_id
                    q.append((id, hop + 1))
        except:
            pass


# In[66]:


for item_id in tqdm(graph.keys()):
    if visited[item_id] == NON_CLUSTER_ID and len(graph[item_id]) > 0:
        BFS(item_id, CLUSTER_ID)
        CLUSTER_ID += 1


# In[67]:


item_["cls_id"] = item_.item.map(visited)


# In[68]:


item_.head(1)


# In[69]:


item_.loc[item_.preprocessed_title.str.contains("etc|ETC"), "cls_id"] = -1


# In[70]:


cnt  = 0
for k, v in Counter(visited.values()).items():
    if v == 1:
        cnt += 1
cnt


# In[71]:


not_clustered_item = []
c = Counter(visited.values())
for k, v in visited.items():
    if c[v] == 1:
        not_clustered_item.append(k)
len(not_clustered_item)


# In[72]:


for item_id in not_clustered_item:
    # 자신과 이웃인 item의 cluster_group을 찾아서 바꿔주기
    group_candidates = []
    for neighbor_item_id in graph[item_id]:
        try:
            group_candidates.append(visited[neighbor_item_id])
        except:
            pass
    CNT = Counter(group_candidates)
    if len(CNT) >= 2:
        tmp = CNT.most_common(2)
        if tmp[0][0] == -1: # 가장 빈도 높은 클러스터가 클러스터링이 안되어있을 때 : -1
            final_cls_id = tmp[1][0] # 나머지 모든 item을 2등 최빈 클러스터로 묶어줌
            for neighbor_item_id in graph[item_id] + [item_id]:
                visited[neighbor_item_id] = final_cls_id
        else:
            final_cls_id = tmp[0][0] # 나머지 모든 item을 1등 최빈 클러스터로 묶어줌
            for neighbor_item_id in graph[item_id] + [item_id]:
                visited[neighbor_item_id] = final_cls_id
    elif len(CNT) == 1:
        tmp = CNT.most_common(1)
        if tmp[0][0] == -1: # 클러스터가 클러스터링이 안되어있을 때 : -1
            final_cls_id = tmp[1][0] # 나머지 모든 item을 not_clustered_item의 클러스터로 묶어줌
            for neighbor_item_id in graph[item_id] + [item_id]:
                visited[neighbor_item_id] = final_cls_id
        else:
            final_cls_id = tmp[0][0] # 나머지 모든 item을 1등 최빈 클러스터로 묶어줌
            for neighbor_item_id in graph[item_id] + [item_id]:
                visited[neighbor_item_id] = final_cls_id
    else: # 이웃들이 모두 잘려나간 카테고리에 포함되는 경우 -> -1로 처리
        visited[item_id] = -1


# In[73]:


not_clustered_item = []
c = Counter(visited.values())
for k, v in visited.items():
    if c[v] == 1:
        not_clustered_item.append(k)
len(not_clustered_item)


# In[74]:


for item_id in not_clustered_item:
    visited[item_id] = -1


# In[75]:


not_clustered_item = []
c = Counter(visited.values())
for k, v in visited.items():
    if c[v] == 1:
        not_clustered_item.append(k)
len(not_clustered_item)


# In[76]:


# 모든 클러스터를 처리함


# In[77]:


len(visited)


# In[78]:


len(item_)


# In[79]:


# 개수 차이가 나는 이유는 visited의 일부가 잘려나간 카테고리에 포함되기 때문
item_["cls_id"] = item_.item.map(visited)


# In[80]:


len(item_[item_.cls_id != -1])
# 전체 88634 item 중에 25039개에 해당하는 아이템을 클러스터링함. (28.2%)


# In[81]:


item_.sample(10)


# In[82]:


type(item_[item_.cls_id!=-1].iloc[0]["similar_item_list"])


# In[83]:


import numpy as np

item_["similar_item_list"] = item_["similar_item_list"].apply(lambda x: np.NaN if len(x) == 0 else "|".join(map(str,x)))


# In[84]:


item_[item_.cls_id!=-1]


# In[85]:


item_.drop(columns=["similarity_list"], inplace=True)


# In[86]:


item_.head(1)


# In[87]:


from time import time

now = str(round(time()))[5:]
item_.to_csv(f"clustered_item_{now}.csv", index=False)


# In[ ]:





# In[ ]:




