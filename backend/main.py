from fastapi import FastAPI, Form, Request, HTTPException
from inference.predict import inference
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware
import json
from service.item import card_house, get_item, get_house_id_with_member_email, get_signup_info, get_random_card
from service.user import check_existing_user, save_db

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


################ Backend ################
import pandas as pd

df = pd.read_csv('data/item.csv')


def type_to_json(df):
    import random
    num = random.sample(range(0,len(df)),6)
    topk = df.loc[num]
    topk = topk[['item','title','seller','price','image']]
    topk_json = topk.to_json(orient="records")
    topk_json = json.loads(topk_json)
    return topk_json

@app.get('/')
async def initial_main_page(): 
    # 모델 결과 Top-K
    # TODO: DB로 부터 초기 페이지 값 불러오기
    return type_to_json(df)


# 로그인 했을 때 메인 페이지
@app.post('/{user_id}')
async def main_page_with_user(
    user_id: str, # index -> user_id
):
   # id -> 개인별 추천상품
    model_result = inference(user_id) 
    item_infos = get_item(model_result)
    
    return user_id, item_infos


# 존재하는 아이디 확인 후 정보 return
@app.get('/login/{member_email}')
async def user_data(member_email: str):
    if check_existing_user(member_email):
        result =  get_house_id_with_member_email(member_email) # result = house_id
        info = []
        return get_item(result)
        #     info+=[get_item(r)]
        # return {member_email, info} # -> member_email, item들
         
    
# db update
class UpdateDBIn(BaseModel):
    user_id: str
    house_id : int
    selected_img_arr: str
    item_id : int
    seller : str
    furniture_name : str
    price : float
    img_url : str
    
    class Config:
        orm_mode = True
        
        
# @app.get('/signup')
# async def get_card_image():
#     '''
#     1. house 테이블에서 스타일별로 5개씩 추출
#     2. json 형태로 리턴
#     '''
#     signup_info = get_signup_info()
#     return get_random_card(signup_info)   

@app.get('/signup/{member_email}')
async def signup(member_email:str) -> list:
    if check_existing_user(member_email):
        return "User already exists."
    else:
        '''
        1. house 테이블에서 스타일별로 5개씩 추출
        2. json 형태로 저장
        3. card img url 형태로 리턴
        '''
        signup_info = get_signup_info()
        house_list = get_random_card(signup_info)
        return [url.get('card_img_url') for url in house_list]


async def update_db(selected_house_list:list, member_email:str):
    
    card_info = {}
    for c in selected_house_list: # c : card_img_url
        card_info[member_email]=card_house(c)  # member_email : house_id
    
    # TODO : 정보 DB저장하기
    return save_db(member_email, card_info)

'''
1. 디테일 페이지에서 보여줄 내용? -> 가격, 이미지링크, 이름, 판매처
2. 회원가입할때 집들이 이미지 5개씩 (스타일별로)
3. 회원가입때 이메일 받아옴
'''
