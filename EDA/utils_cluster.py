import pandas as pd
from tqdm import tqdm
from collections import deque, Counter

global del_list
CLUSTER_ID = 0
NON_CLUSTER_ID = -1

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

def get_del_list():
    global del_list
    return del_list

def get_cluster_by_BFS(df:pd.DataFrame, args):
    def BFS(dfid:int, cls_id):
        q = deque([(dfid, 0)])
        visited[dfid] = cls_id
        while(q):
            node, hop = q.popleft()
            if hop > args.limit_depth:
                return
            try:
                for id in graph[node]:
                    if visited[id] == -1: # 아직 클러스터링되지 않은 아이템인 경우
                        visited[id] = cls_id
                        q.append((id, hop + 1))
            except:
                pass
    graph = {dfid:sim_items for dfid, sim_items in zip(df.item, df.similar_dflist)}
    visited = {dfid:NON_CLUSTER_ID for dfid in df.item}
    # 양방향 그래프로 만들기
    for dfid, sim_items in graph.items():
        for dfid_ in sim_items:
            try:
                graph[dfid_].append(dfid)
            except:
                pass
    for dfid in tqdm(graph.keys()):
        if visited[dfid] == NON_CLUSTER_ID and len(graph[dfid]) > 0:
            BFS(dfid, CLUSTER_ID)
            CLUSTER_ID += 1
    
    df["cls_id"] = df.item.map(visited)
    
    df.loc[df.preprocessed_title.str.contains("etc|ETC"), "cls_id"] = -1
    
    not_clustered_item = []
    c = Counter(visited.values())
    for k, v in visited.items():
        if c[v] == 1:
            not_clustered_item.append(k)
    
    for dfid in not_clustered_item:
        # 자신과 이웃인 item의 cluster_group을 찾아서 바꿔주기
        group_candidates = []
        for neighbor_dfid in graph[dfid]:
            try:
                group_candidates.append(visited[neighbor_dfid])
            except:
                pass
        CNT = Counter(group_candidates)
        if len(CNT) >= 2:
            tmp = CNT.most_common(2)
            if tmp[0][0] == -1: # 가장 빈도 높은 클러스터가 클러스터링이 안되어있을 때 : -1
                final_cls_id = tmp[1][0] # 나머지 모든 item을 2등 최빈 클러스터로 묶어줌
                for neighbor_dfid in graph[dfid] + [dfid]:
                    visited[neighbor_dfid] = final_cls_id
            else:
                final_cls_id = tmp[0][0] # 나머지 모든 item을 1등 최빈 클러스터로 묶어줌
                for neighbor_dfid in graph[dfid] + [dfid]:
                    visited[neighbor_dfid] = final_cls_id
        elif len(CNT) == 1:
            tmp = CNT.most_common(1)
            if tmp[0][0] == -1: # 클러스터가 클러스터링이 안되어있을 때 : -1
                final_cls_id = tmp[1][0] # 나머지 모든 item을 not_clustered_item의 클러스터로 묶어줌
                for neighbor_dfid in graph[dfid] + [dfid]:
                    visited[neighbor_dfid] = final_cls_id
            else:
                final_cls_id = tmp[0][0] # 나머지 모든 item을 1등 최빈 클러스터로 묶어줌
                for neighbor_dfid in graph[dfid] + [dfid]:
                    visited[neighbor_dfid] = final_cls_id
        else: # 이웃들이 모두 잘려나간 카테고리에 포함되는 경우 -> -1로 처리
            visited[dfid] = -1
    not_clustered_item = []
    c = Counter(visited.values())
    for k, v in visited.items():
        if c[v] == 1:
            not_clustered_item.append(k)
    for dfid in not_clustered_item:
        visited[dfid] = -1

    # 개수 차이가 나는 이유는 visited의 일부가 잘려나간 카테고리에 포함되기 때문
    df["cls_id"] = df.item.map(visited)
    return df