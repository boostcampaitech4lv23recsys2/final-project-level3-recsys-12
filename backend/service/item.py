from db.models import Item, House, Member, HouseItem, MemberPrefer
from db.db_connect import Database
from sqlalchemy import select
import random
import pandas as pd

database = Database()

def random_item():
    with database.session_maker() as session:
        stmt = select(Item).order_by(Item.rating.desc())
        return session.execute(stmt).fetchmany(10)
    
def get_item(item_ids):
    # Read data
    item_infos = {}
    for item in item_ids:
        print('item',item)
        with database.session_maker() as session:
            stmt = select(HouseItem).where(HouseItem.item_id == item)
            item_info = session.execute(stmt).fetchall()
            item_infos[item] = item_info

    return item_ids, item_infos

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

def get_item_info(item_id : int):
    with database.session_maker() as session:
        stmt = select(Item.item_id, Item.title, Item.price, Item.image, Item.seller, Item.predict_price).where(Item.item_id==item_id)
        data = session.execute(stmt).fetchall()
        return data
    
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

def get_signup_info():
    with database.session_maker() as session:
        stmt = select(House.house_id, House.style, House.card_img_url).where(House.card_space == "거실")
        signup_infos = session.execute(stmt).fetchall()
    return signup_infos


def get_random_card(signup_info):
    import json
    SAMPLE_NUM = 5
    signup_info_df = pd.DataFrame(signup_info, columns=["house_id", "style", "card_img_url"])
    cats = set()
    for sets in signup_info_df["style"].apply(lambda x: set(x.split(", "))):
        cats = cats.union(sets)
    for cat_name in filter(lambda x: x, cats):
        signup_info_df[cat_name] = signup_info_df["style"].str.contains(cat_name).astype(int)
    # print(signup_info_df)
    house_id_list = []
    for category in filter(lambda x: x, cats):
        house_id_list.append(random.sample(list(signup_info_df[signup_info_df[category] == 1].index), SAMPLE_NUM))
    
    house_id_list = sum(house_id_list, [])    # 2차원 배열 1차원으로 펴줌

    return_signup_info = signup_info_df.iloc[house_id_list]
    
    return_signup_info = return_signup_info[["house_id", "card_img_url", "style"]]
    return_signup_info = return_signup_info.sample(frac=1)
    return_signup_info = return_signup_info.to_json(orient="records")
    return_signup_info = json.loads(return_signup_info)

    return return_signup_info

def check_is_prefer(member_email, item_id):
    with database.session_maker() as session:
        stmt = f"select * from member_prefer where member_email='{member_email}' and item_id='{item_id}'"
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
        stmt = f"delete from member_prefer where member_email='{member_email}' and item_id='{item_id}'"
        if not stmt:
            return "failure"
        session.execute(stmt)
        session.commit()
        return "success"