from fastapi import FastAPI, Form, Request, HTTPException
from inference.predict import inference
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

import json
from datetime import timedelta, datetime

from jose import jwt
import secrets
import requests
import os
import yaml

from service.user import *
from service.item import *

from inference.predict import Model
import pandas as pd

app = FastAPI()


################ Slack 연결 ################
SECRET_FILE = os.path.join('../secrets.yaml')
with open(SECRET_FILE) as fp:
    secret_file = yaml.load(fp, yaml.FullLoader)
SLACK = secret_file["SLACK"]


# send slack message
def post_slack_message(text):
    response = requests.post("https://slack.com/api/chat.postMessage",
                            headers={"Authorization": "Bearer " + SLACK["token"]},
                            data={"channel": "#serverlog", "text": text}
                            )
    print(response)


############ first setting ############
df_for_model = pd.read_csv("data/train.csv").groupby("house").filter(lambda x: len(x) >= 15)

with open("inference/model.yaml") as f:
    model_info = yaml.load(f, Loader=yaml.FullLoader)

MODEL = Model(model_info, df_for_model)


################ Front 연결 ################
origins = [
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"))

@app.get("/")
def index():
    return FileResponse("../frontend/dist/index.html")

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
    import random
    model_result = [result[0] for result in get_inference_result(member_email)]
    
    # random.shuffle(model_result)
    item_list = []
    for item_id in model_result[:]:
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


class Image(BaseModel):
    member_email :str
    selected_house_id : str
    
@app.post('/member')
async def image(item:Image):
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
    if len(check_is_prefer(member_email, item_id)) != 0:
        return "failure"
    return insert_member_prefer(member_email, item_id)

@app.delete('/delete-prefer/{member_email}/{item_id}')
async def delete_prefer_data(member_email: str, item_id: int):
    return delete_member_prefer(member_email, item_id)


class InferenceResult(BaseModel):
    member_email: str
@app.post('/insert-inference-result')
async def insert_inference_result(inference_result: InferenceResult, description="화원가입할 때 inference"):
    user_prefered_item = get_inference_input(inference_result.member_email)  # 모델에 넣을 input list(item_id_list)
    print(user_prefered_item)
    model_result = inference(user_prefered_item, MODEL)    # 모델 인퍼런스
    return create_inference(inference_result.member_email, model_result)

# 디비에 있는 좋아요를 꺼내와서 (아이템 아이디) inference input에 

class UpdataInferenceResult(BaseModel):
    member_email: str
    
@app.post('/update-inference-result')
async def insert_inference_result(update_inference: UpdataInferenceResult, description="좋아요 반영 inference"):
    member_prefer = get_item_prefer(update_inference.member_email) # 
    inference_result = [result[0] for result in get_inference_result(update_inference.member_email)]
    update_list = member_prefer+inference_result
    model_result = inference(update_list, MODEL)
    delete_inference(update_inference.member_email)
    difference = set(model_result) - set(inference_result)
    intersection = set(model_result) & set(inference_result)
    create_inference(update_inference.member_email, model_result),
    if difference:
        return {
            "new_item":get_item(difference),
            "inter": get_item(intersection)
        }
    else:
        return "Already Update"