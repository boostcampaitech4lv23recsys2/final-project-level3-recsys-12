from fastapi import FastAPI, Form, Request, HTTPException
from model.predict import inference
from pydantic import BaseModel, Field

app = FastAPI()

# TODO : 로그인 구현, 상품 구현
# TODO : 로그인(login) = Request
# TODO : 상품(product) = 가구 추천 결과

import pandas as pd
df = pd.read_csv('../item_v1.csv')

fake_users = {
    "Boostcamp" : {
        'item_id' : df.item[0:6].tolist(),
        'furniture_name' : df.title[0:6].tolist(),
        'seller' : df.seller[0:6].tolist(),
        'price' : list(map(str,df.price[0:6].tolist())),
        "image_url" : df.image[0:6].tolist(),},
    "AI" : {
        'item_id' : df.item[6:12].tolist(),
        'furniture_name' : df.title[6:12].tolist(),
        'seller' : df.seller[6:12].tolist(),
        'price' : list(map(str,df.price[6:12].tolist())),
        "image_url" : df.image[6:12].tolist(),},
    "Camp" : {
        'item_id' : df.item[12:18].tolist(),
        'furniture_name' : df.title[12:18].tolist(),
        'seller' : df.seller[12:18].tolist(),
        'price' : list(map(str,df.price[12:18].tolist())),
        "image_url" : df.image[12:18].tolist(),}
}

# 200 -> 성공
# 400 -> 실패
SUCCESS_CODE = 200
FAIL_CODE = 400

# 순서 -> item_id, 가구명, 가구파는 곳, 가격, 이미지 url
@app.get('/')
async def initial_main_page(): #item_id, 가구명, 가구파는 곳, 가격, 이미지 url
    # 모델 결과 Top-K
    # TODO: DB로 부터 초기 페이지 값 불러오기

    return {"item_id" : df.item[18:24].tolist(), "funiture_name" : df.title[18:24].tolist(), "seller" : df.seller[18:24].tolist(),\
        "price" : list(map(str,df.price[18:24].tolist())), "img_url" : df.image[18:24].tolist()} # user별 추천 item K개


# 로그인 했을 때 메인 페이지
@app.get('/{user_id}') #가구명, 가구파는 곳, 가격, 이미지 url, item_id
async def main_page_with_user(
    user_id: str, # index -> user_id
    item_id : int,
    furniture_name : str,
    seller : str,
    price : float,
    img_url : str,
):
   # id -> 개인별 추천상품
    return inference(user_id, item_id, furniture_name, seller, price, img_url) # top-k list

# login 했을 때 존재하는 아이디인지 확인
@app.get('/login/{user_id}')
async def login_status(user_id: str):
    if user_id in fake_users:
        return SUCCESS_CODE
    else:
        return FAIL_CODE

# 존재하는 아이디 확인 후 정보 return
@app.get('/login/{user_id}')
async def user_data(user_id: str): #가구명, 가구파는 곳, 가격, 이미지 url, item_id
    return fake_users[user_id]

# 회원가입
@app.get('/register/{user_id}')
async def duplicate_name_check(user_id : str):
    if user_id in fake_users:
        return FAIL_CODE
    else:
        return SUCCESS_CODE

# db update
@app.get('/register/{user_id}')
async def update_db(user_id: str,
    item_id : int,
    furniture_name : str,
    seller : str,
    price : float,
    img_url : str,):
    new_user = {user_id : {"item_id": item_id, "furniture_name":furniture_name,"seller" : seller, "price" : price, "img_url":img_url}}
    fake_users.update(new_user)
    return fake_users
        
# # 개인 페이지
# @app.get('/users/{user_id}')
# def get_user_item(user_id):
#     return user_id, fake_users[user_id]

# @app.get('/users/{product}/')
# def product_image(user_id : str):
#     #item = db.query(models.items).filter(models.items.item_id == item_id)
#     item = fake_users[user_id]
#     #item.image_url = f"/main/furniture_image/{item.item_id}.png" 
#     return item

