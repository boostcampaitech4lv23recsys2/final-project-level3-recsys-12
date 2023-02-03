from db.models import *

from db.db_connect import Database
from sqlalchemy import select
import random
import pandas as pd

database = Database()

def random_item():
    with database.session_maker() as session:
        stmt = f"select * from item where rating>='4.7' and price!='' and review>='5'"
        return session.execute(stmt).fetchall()
    
def get_item(item_ids):
    # Read data
    item_infos = []
    for item in item_ids:
        with database.session_maker() as session:
            stmt = select(Item).where(Item.item_id == item)
            item_info = session.execute(stmt).fetchall()
            item_infos.append([col[0] for col in item_info][0])
    
    return item_infos

def house_get_item(item_ids):
    # Read data
    item_infos = {}
    for house in item_ids:
        with database.session_maker() as session:
            stmt = select(HouseItem).where(HouseItem.house_id == house)
            item_info = session.execute(stmt).fetchall()
            item_infos[house] = item_info

    return item_ids, item_infos

def card_house(card_img_url : str) -> list:
    with database.session_maker() as session:
        stmt = select(House).where(House.card_img_url==card_img_url)
        data = session.execute(stmt).fetchall()
        return [col[0] for col in data][0].house_id

    
def get_item_info_all(item_id : int):
    with database.session_maker() as session:
        # stmt = select(Item).where(Item.item_id==item_id)
        # stmt = f"select * from item where item_id={item_id} and rating!='0.0' and review>'1'"
        # stmt = "select * from item where item_id='%d' and rating!='0.0' and review>'1'" % (item_id)
        stmt = "select * from item where item_id='%d'" % item_id
        data = session.execute(stmt).fetchall()
        return data
        # return [col[0] for col in data]
    
def get_house_id_with_member_email(member_email:str) -> str:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email==member_email)
        data = session.execute(stmt).fetchall()
        house_id_list = []
        for user_info in data:
            house_id_list.append(user_info[0].house_id)
        return house_id_list    # 유저가 선호한 house_id 리스트
        # return [col[0] for col in data].house_id

def get_item_list_by_house_id(house_id):
    with database.session_maker() as session:
        stmt = select(HouseItem.item_id).where(HouseItem.house_id==house_id)
        data = session.execute(stmt).fetchall()
        return data
    
def get_inference_input(member_email):
    user_prefered_house = get_house_id_with_member_email(member_email)
    user_prefered_item_json = []
    for house_id in user_prefered_house:
        user_prefered_item_json.append(get_item_list_by_house_id(house_id))
    user_prefered_item_json = sum(user_prefered_item_json, [])  # 2차원 배열 -> 1차원
    user_prefered_item = [] # 모델에 들어갈 input item_id로 이루어진 1차원 list
    for item_id in user_prefered_item_json:
        user_prefered_item.append(item_id[0])
    return user_prefered_item

def get_house_id_with_member_email(member_email:str) -> str:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email==member_email)
        data = session.execute(stmt).fetchall()
        return [col.Member.house_id for col in data]

################################### 랜덤 5개씩 스타일별로 ###################################
# def get_signup_info():
#     with database.session_maker() as session:
#         stmt = select(House.house_id, House.style, House.card_img_url).where(House.card_space == "거실")
#         signup_infos = session.execute(stmt).fetchall()
#     return signup_infos


# def get_random_card(signup_info):
#     import json
#     SAMPLE_NUM = 5
#     signup_info_df = pd.DataFrame(signup_info, columns=["house_id", "style", "card_img_url"])
#     cats = set()
#     for sets in signup_info_df["style"].apply(lambda x: set(x.split(", "))):
#         cats = cats.union(sets)
#     for cat_name in filter(lambda x: x, cats):
#         signup_info_df[cat_name] = signup_info_df["style"].str.contains(cat_name).astype(int)
#     house_id_list = []
#     for Style in filter(lambda x: x, cats):
#         Style_list = random.sample(list(signup_info_df[signup_info_df[Style] == 1].index), SAMPLE_NUM)
#         house_id_list+=Style_list
#         cnt = check_duplicates(house_id_list, Style_list) # 중복체크하기
#         if cnt%5!=0:
#             while (cnt%5)>0: # 다시 추천해 준 것에서 다시 중복이 발생할 경우
#                 add = (random.sample(list(signup_info_df[signup_info_df[Style] == 1].index), 5-cnt%5))
#                 house_id_list+=add
#                 cnt = check_duplicates(house_id_list, Style_list)

#     house_id_list = list(set(house_id_list))
#     return_signup_info = signup_info_df.iloc[house_id_list]
    
#     return_signup_info = return_signup_info[["house_id", "card_img_url", "style"]]
#     return_signup_info = return_signup_info.sample(frac=1)
#     return_signup_info = return_signup_info.to_json(orient="records")
#     return_signup_info = json.loads(return_signup_info)
    
#     return return_signup_info

# def check_duplicates(seq1, seq2):
#     duplicates = [x for i, x in enumerate(seq1) if i != seq1.index(x)]  # 중복된 아이템 확인
#     return len(set(seq1) - set((seq2)))
###############################################################################################

################################### tree ###################################
from sqlalchemy import func

def get_house_style():
    Style = []
    with database.session_maker() as session:
        stmt = select(House.house_id, House.style)
        data = session.execute(stmt).fetchall()
        data = [col for col in data]
    
    Style = (sum([col.style.split(", ") for col in data],[]))
    Style = list(set(Style))
    house_id_list = []
    for cate_name in Style[1:]:
        with database.session_maker() as session:
            stmt = select(House.house_id, House.style).where(House.style == cate_name).limit(1)
            # stmt = select(House.house_id, House.style).where(House.style == cate_name).order_by(func.random()).limit(1)
            data = session.execute(stmt).fetchall()
            house = [col.house_id for col in data]
            house_id_list+=house
    return house_id_list # 초기 11개
'''
1. 초기 이미지 11개 보여주기
2. 유저가 선택한 카드번호 리스트 모델에 넣기
3. return 유저가 선택한 카드 url 리스트 
'''
def get_card(house_id_list): # 카드 조건 걸기

    card_infos = []
    for house in house_id_list:
        with database.session_maker() as session:
            stmt = select(Card).where(Card.house_id==house).where(Card.img_space =='거실' and Card.is_human==0).order_by(func.random()).limit(1)
            data = session.execute(stmt).fetchall()
            card = [col.Card.card_id for col in data]
            card_infos += card

    return card_infos

def get_card_info(card_id_list): # 카드 id를 가지고 house_id 찾긴
    card_infos = []
    for card_id in card_id_list:
        with database.session_maker() as session:
            stmt = select(Card).where(Card.card_id==card_id)
            data = session.execute(stmt).fetchall()
            house = [col.Card for col in data]
            card_infos += house

    return card_infos

'''

'''
###############################################################################################


def check_is_prefer(member_email, item_id):
    with database.session_maker() as session:
        # stmt = f"select * from member_prefer where member_email='{member_email}' and item_id='{item_id}'"
        stmt = "select * from member_prefer where member_email='%s' and item_id='%d'"  % (member_email, item_id)
        # stmt = select(MemberPrefer).where(member_email==member_email and item_id==item_id)
        is_prefer = session.execute(stmt).fetchall()
        return is_prefer

def insert_member_prefer(member_email, item_id):
    with database.session_maker() as session:
        stmt = MemberPrefer(member_email=member_email, item_id=item_id)
        session.add(stmt)
        session.commit()
        return "success"

def delete_member_prefer(member_email, item_id):
    with database.session_maker() as session:
        # stmt = f"delete from member_prefer where member_email='{member_email}' and item_id='{item_id}'"
        stmt = "delete from member_prefer where member_email='%s' and item_id='%d'" % (member_email, item_id)
        if not stmt:
            return "failure"
        session.execute(stmt)
        session.commit()
        return "success"
    

def get_item_prefer(member_email : str):
    with database.session_maker() as session:
        stmt = select(MemberPrefer).where(MemberPrefer.member_email==member_email)
        data = session.execute(stmt).fetchall()
        return [col[0].item_id for col in data]
    
def get_item_info_prefer(item_ids : list):
    item_infos = {}
    for item in item_ids:
        with database.session_maker() as session:
            stmt = select(Item).where(Item.item_id == item)
            item_info = session.execute(stmt).fetchall()
            item_infos[item] = item_info[0].Item
    return [(v, k)[0] for k, v in item_infos.items()]

def get_inference_result(member_email):
    with database.session_maker() as session:
        # stmt = f"select item_id from inference_result where member_email='{member_email}'"
        stmt = "select item_id from inference_result where member_email='%s'" % (member_email)
        return session.execute(stmt).fetchall()

def delete_inference(member_email):
    with database.session_maker() as session:
        # stmt = f"delete from inference_result where member_email='{member_email}'"
        stmt = "delete from inference_result where member_email='%s'" % (member_email)
        if not stmt:
            return "failure"
        session.execute(stmt)
        session.commit()
        return "success"

def get_item_cluster(item_id):
    with database.session_maker() as session:
        stmt = select(ClusterItem.cluster_id).where(ClusterItem.item_id==item_id)
        data = session.execute(stmt).fetchall()
        return data
        
def get_clusters(cluster_id, item_id):
    with database.session_maker() as session:
        stmt = select(ClusterItem.item_id).where(ClusterItem.cluster_id==cluster_id).where(ClusterItem.item_id!=item_id)
        return session.execute(stmt).fetchall()

def get_same_series_item(item_id, category, seller):
    with database.session_maker() as session:
        stmt = select(Item.item_id).where(Item.item_id!=item_id).where(Item.category==category).where(Item.seller==seller)
        return session.execute(stmt).fetchall()

def get_popular_item(item_id, category):
    # 현재 인기도 기준 : 별점 4.7 이상, review 수 5개 이상
    with database.session_maker() as session:
        stmt = select(Item.item_id).where(Item.item_id!=item_id).where(Item.category==category).where(Item.rating>=4.7).where(Item.review>=5)
        return session.execute(stmt).fetchall()