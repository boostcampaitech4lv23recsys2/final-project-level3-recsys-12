from db.models import Item,  House, Member, HouseItem
from db.db_connect import Database
from sqlalchemy import select
from inference.predict import inference
import random
import pandas as pd

database = Database()


# def get_items(item_ids):
#     # Read data
#     item_infos = {}
#     for item_id in item_ids:
#         with database.session_maker() as session:
#             stmt = select(House).where(House.item_id == item_id) # Statement -> DB Query를 의미
#             item_info = session.execute(stmt).fetchall()
#             item_infos[item_id] = item_info
#     return item_ids, item_infos

def card_house(card_url : int) -> list:
    with database.session_maker() as session:
        stmt = select(House).where(House.card_url==card_url)
        data = session.execute(stmt).fetchall()
        return [col[0] for col in data][0].house_id

def get_item(house_id : int) -> list:
    with database.session_maker() as session:
        stmt = select(HouseItem).where(HouseItem.house_id==house_id)
        data = session.execute(stmt).fetchall()
        try:
            return [col[0] for col in data][0].item_id
        except:
            return "No exist item"
    
def get_house_id_with_member_email(member_email:str) -> str:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email==member_email)
        data = session.execute(stmt).fetchall()
        return [col[0] for col in data][0].house_id


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
