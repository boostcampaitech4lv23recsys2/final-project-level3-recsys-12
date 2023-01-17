from fastapi import FastAPI, Form, Request, HTTPException
from inference.predict import inference
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import json
from service.item import get_item
from service.user import get_user_info, insert_user

app = FastAPI()

# TODO : 로그인 구현, 상품 구현
# TODO : 로그인(login) = Request
# TODO : 상품(product) = 가구 추천 결과

################ Front 연결 ################
origins = [
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

################ DB 연결 ################



################ Backend ################
import pandas as pd

df = pd.read_csv('item_v1.csv')

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

# 200 -> 성공
# 400 -> 실패
SUCCESS_CODE = 200
FAIL_CODE = 400

def type_to_json(df):
    import random
    num = random.randint(0,len(df)-6)
    topk = df.loc[num:num+6]
    topk = topk[['item','title','seller','price','image']]
    topk_json = topk.to_json(orient="records")
    topk_json = json.loads(topk_json)
    return topk_json

# 순서 -> item_id, 가구명, 가구파는 곳, 가격, 이미지 url
@app.get('/')
async def initial_main_page(): #item_id, 가구명, 가구파는 곳, 가격, 이미지 url
    # 모델 결과 Top-K
    # TODO: DB로 부터 초기 페이지 값 불러오기
    return type_to_json(df)


# 로그인 했을 때 메인 페이지
@app.get('/{user_id}') #가구명, 가구파는 곳, 가격, 이미지 url, item_id
async def main_page_with_user(
    user_id: str, # index -> user_id
):
   # id -> 개인별 추천상품
    model_result = inference(user_id) 
    item_infos = get_item(model_result)
    
    return user_id, item_infos


# 존재하는 아이디 확인 후 정보 return
@app.get('/login/{user_id}')
async def user_data(user_id: str): #가구명, 가구파는 곳, 가격, 이미지 url, item_id
    return get_user_info(user_id)


# db update
class UpdateDBIn(BaseModel):
    user_id: str
    selected_img_arr: str
    item_id : int
    seller : str
    furniture_name : str
    price : float
    img_url : str

@app.post('/register/{user_id}')
async def update_db(update_db_in: UpdateDBIn) -> str:
    # 회원가입할 때 선택한 집들이 이미지 list를 받을텐데 DB에 list
    '''
    1. 해당 데이터에 대한 별도의 테이블을 구성하고 쿼리문의 조인을 통해 DTO를 구성하는 방법이다.
    2. 그냥 배열 형태의 데이터를 통째로 String으로 변환 후 DB에 저장하고 꺼내올 때는 String을 파싱하여 List에 담아서 보내는 것이다. -> lst = '[1,2,3,4]'
    '''
    return insert_user(update_db_in.user_id, update_db_in.selected_img_arr)
        
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

