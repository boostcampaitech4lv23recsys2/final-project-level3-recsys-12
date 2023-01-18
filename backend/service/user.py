# from main import fake_users
from db.models import Item
from db.models import House
from sqlalchemy.orm import Session
from pydantic import BaseModel, validator, EmailStr
import pandas as pd

df = pd.read_csv('data/item_v1.csv')

fake_users = {
    "Boostcamp" : {
        'item_id' : f"{df.item[0:6].tolist()}",
        'furniture_name' : f"{df.title[0:6].tolist()}",
        'seller' : f"{df.seller[0:6].tolist()}",
        'price' : f"{list(map(str,df.price[0:6].tolist()))}",
        "image_url" : f"{df.image[0:6].tolist()}"},
    "AI" : {
        'item_id' : f"{df.item[6:12].tolist()}",
        'furniture_name' : f"{df.title[6:12].tolist()}",
        'seller' : f"{df.seller[6:12].tolist()}",
        'price' : f"{list(map(str,df.price[6:12].tolist()))}",
        "image_url" : f"{df.image[6:12].tolist()}"},
    "Camp" : {
        'item_id' : f"{df.item[12:18].tolist()}",
        'furniture_name' : f"{df.title[12:18].tolist()}",
        'seller' : f"{df.seller[12:18].tolist()}",
        'price' : f"{list(map(str,df.price[12:18].tolist()))}",
        "image_url" : f"{df.image[12:18].tolist()}"},
}

# def get_user_info(user_id):
#     if user_id in fake_users:
#         return fake_users[user_id]
#     else:
#         return "No user in database"
    
# def check_duplicated_id(user_id):
#     if user_id in fake_users:
#         return False 
#     else:
#         return True
    
# def insert_user(user_id:str, selected_img:str):
#     # 유저 아이디 있는지 체크
#     if check_duplicated_id(user_id):
#         # 있으면 중복 되었다고 반환
#         return "User already exists."

#     else:
#         # 없으면 DB에 유저 정보 저장
#         new_user = {user_id : selected_img}
#         fake_users.update(new_user)
#         return "OK update complete"


        
# def get_existing_user(db:Session, house_id: int):
#     return db.query(House).filter(House.house_id==house_id).first()
from db.db_connect import Database
from db.models import House
from sqlalchemy import select

database = Database()

def get_existing_user(house_id: int):
    with database.session_maker() as session:
        stmt = select(House).where(House.house_id == house_id) # Statement -> DB Query를 의미

        return session.execute(stmt).fetchall()

def check_duplicated_id(db:Session, user_id):
    if get_existing_user(user_id):
        return False 
    else:
        return True
    
def insert_user(db:Session, user_id:str, selected_img:str):
    # 유저 아이디 있는지 체크
    if check_duplicated_id(user_id):
        # 있으면 중복 되었다고 반환
        return "User already exists."

    else:
    #     # 없으면 DB에 유저 정보 저장
    #     new_user = {user_id : selected_img}
    #     # fake_users.update(new_user)
    #     db.add(new_user)
        return "OK update complete"