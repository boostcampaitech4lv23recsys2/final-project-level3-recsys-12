from fastapi import FastAPI, Form, Request, HTTPException
from inference.predict import inference
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import timedelta, datetime

from jose import jwt
import secrets
from service.item import get_house_id_with_member_email, get_random_card, get_signup_info, random_item, get_item
from service.user import check_existing_user, create_member
from service.item import get_item_info, get_item_list_by_house_id, get_inference_input

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

@app.get('/')
async def initial_main_page(description='비로그인 초기 페이지에 랜덤으로 아이템을 출력하는 부분입니다.'):
    return random_item()

# 로그인 했을 때 메인 페이지
@app.get('/{member_email}')
async def main_page_with_user(
    member_email: str
):
    # id -> 개인별 추천상품
    user_prefered_item = get_inference_input(member_email)  # 모델에 넣을 input list(item_id_list)
    model_result = inference(user_prefered_item)    # 모델 인퍼런스
    
    item_list = []
    for item_id in model_result:
        item_list.append(get_item_info(item_id))   # item
    item_list = sum(item_list, [])
    return item_list


ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"

# 존재하는 아이디 확인 후 정보 return
@app.get('/login/{member_email}')
async def login(member_email: str):
    # 로그인 시 해당 이메일이 회원인지 확인
    if not check_existing_user(member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email doesn\'t exist'"))
    data = {
        "sub": member_email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "member_email": member_email
    }
         
    
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

@app.get('/signup')
async def get_card_image():
    '''
    1. house 테이블에서 스타일별로 5개씩 추출
    2. json 형태로 리턴
    '''
    signup_info = get_signup_info()
    return get_random_card(signup_info)   
    


@app.get('/signup/{member_email}')
async def signup(member_email:str, discription='회원가입 API입니다.') -> list:
    if check_existing_user(member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email already exist'"))


@app.post('/image_url_save/')
async def image(house_id_list:list, member_email:str):
    return create_member(house_id_list, member_email)

'''
1. 디테일 페이지에서 보여줄 내용? -> 가격, 이미지링크, 이름, 판매처
2. 회원가입할때 집들이 이미지 5개씩 (스타일별로)
3. 회원가입때 이메일 받아옴
'''

'''
get : dict(json)를 받을 수 없음, {} 있을수도 없을수도
post : dict(json)를 받을 수 있음. {}로만 움직임.
'''
