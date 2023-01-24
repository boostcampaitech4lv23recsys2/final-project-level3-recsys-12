from fastapi import FastAPI, Form, Request, HTTPException
from inference.predict import inference
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import timedelta, datetime

from jose import jwt
import secrets
import requests
import os
import yaml

from service.user import check_existing_user, create_member
from service.item import get_random_card, get_signup_info, random_item, get_item_info, get_item_info_all, get_inference_input
from service.item import check_is_prefer, insert_member_prefer, delete_member_prefer, get_item_info, get_item_info_all, get_inference_input

app = FastAPI()


################ Slack 연결 ################
SECRET_FILE = os.path.join('../secrets.yaml')
with open(SECRET_FILE) as fp:
    serects = yaml.load(fp, yaml.FullLoader)
SLACK = serects["SLACK"]

# send slack message
def post_slack_message(text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                            headers={"Authorization": "Bearer " + SLACK["token"]},
                            data={"channel": "#serverlog", "text": text}
                            )
    print(response)


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

@app.get('/home')
async def initial_main_page(description='비로그인 초기 페이지에 랜덤으로 아이템을 출력하는 부분입니다.'):
    """
    rating 높은 순 100개 랜덤으로
    """
    items = random_item()
    item_list = [col.Item for col in items]
    return item_list
    
# 로그인 했을 때 메인 페이지
@app.get('/home/{member_email}')
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

class Login(BaseModel):
    member_email : str
@app.post('/login')
async def login(DB_login : Login):
    # 로그인 시 해당 이메일이 회원인지 확인
    if not check_existing_user(DB_login.member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email doesn't exist'"))
    data = {
        "sub": DB_login.member_email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "member_email": DB_login.member_email
    }
            

@app.get('/card')
async def get_card_image():
    '''
    1. house 테이블에서 스타일별로 5개씩 추출
    2. json 형태로 리턴
    '''
    signup_info = get_signup_info()
    return get_random_card(signup_info)   


class Signup(BaseModel):
    member_email : str

@app.post('/signup')
async def signup(db_signup:Signup, discription='회원가입 API입니다.') -> list:
    if check_existing_user(db_signup.member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email already exist'"))
    else:
        return JSONResponse(status_code=200, content=dict(msg="Success'"))



########## 추가 부분 ############
class Image(BaseModel):
    member_email :str
    selected_house_id : str
@app.post('/member')
async def image(item:Image):
    print("hererere")
    return (create_member(item.member_email, item.selected_house_id))

'''
get : dict(json)를 받을 수 없음, {} 있을수도 없을수도
post : dict(json)를 받을 수 있음. {}로만 움직임.
'''

@app.get('/item/{item_id}')
async def detail(item_id:int):
    item_info = get_item_info_all(item_id)
    return item_info[0].Item


class Mypage(BaseModel):
    member_email : str

@app.post('/mypage')
async def detail(mypage : Mypage):
    from service.item import get_item_prefer, get_item_info_prefer
    item_list = get_item_prefer(mypage.member_email)
    return get_item_info_prefer(item_list)

@app.get('/prefer/{member_email}/{item_id}')
async def is_prefer_item(member_email: str, item_id: int):
    """유저가 item에 좋아요를 눌렀는지 확인

    Args:
        member_email (str): 유저의 이메일
        item_id (int): 아이템 id
    """
    if len(check_is_prefer(member_email, item_id)) == 0:
        return False
    return True

@app.get('/insert-prefer/{member_email}/{item_id}')
async def insert_prefer_data(member_email: str, item_id: int):
    return insert_member_prefer(member_email, item_id)

@app.delete('/delete-prefer/{member_email}/{item_id}')
async def delete_prefer_data(member_email: str, item_id: int):
    return delete_member_prefer(member_email, item_id)