import json
import os
import secrets
from datetime import datetime, timedelta

import pandas as pd
import requests
import yaml
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from inference.predict import Model, inference

# signup inference
from inference.predict_signup import CardVectorizer, Config, ImageUtil
from jose import jwt
from pydantic import BaseModel
from service.item import *
from service.user import *
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

app = FastAPI()


################ Slack 연결 ################
SECRET_FILE = os.path.join("../secrets.yaml")
with open(SECRET_FILE) as fp:
    secret_file = yaml.load(fp, yaml.FullLoader)
SLACK = secret_file["SLACK"]


# send slack message
def post_slack_message(text):
    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer " + SLACK["token"]},
        data={"channel": "#serverlog", "text": text},
    )
    print(response)


############ first setting ############
df_for_model = (
    pd.read_csv("data/train.tsv", sep="\t")
    .groupby("house")
    .filter(lambda x: len(x) >= 15)
)
item_df_for_model = (
    pd.read_csv("data/item.tsv", sep="\t")
)

with open("inference/model.yaml") as f:
    model_info = yaml.load(f, Loader=yaml.FullLoader)

MODEL = Model(model_info, df_for_model, item_df_for_model)

############ Signup Setting ############
card_vector = CardVectorizer(Config())
card_vector.load_data()

################ Front 연결 ################
origins = [
    "http://127.0.0.1:5173"
    # "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###################################배포/개발 환경설정###################################
# 배포 환경에서는 해당 구역을 활성화하면 됩니다
# 개발 환경에서는 해당 구역을 주석처리하면 됩니다
app.mount("/assets", StaticFiles(directory="../frontend/dist/assets"))


@app.get("/")
def index():
    return FileResponse("../frontend/dist/index.html")


#########################################################################################


################ Backend ################
@app.get("/home")
async def initial_main_page(description="비로그인 초기 페이지에 랜덤으로 아이템을 출력하는 부분입니다."):
    """
    rating 높은 순 100개 랜덤으로
    """
    items = random_item()
    random.shuffle(items)
    return items[:50]


# 로그인 했을 때 메인 페이지
@app.get("/home/{member_email}")
async def main_page_with_user(member_email: str):
    import random

    model_result = [result[0] for result in get_inference_result(member_email)]

    # random.shuffle(model_result)
    item_list = []

    for item_id in model_result[:]:
        item = get_item_info_all(item_id)
        item_list.append(item)  # item

    item_list = sum(item_list, [])
    return item_list


ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
SECRET_KEY = secrets.token_hex(32)
ALGORITHM = "HS256"

# 존재하는 아이디 확인 후 정보 return


class Login(BaseModel):
    member_email: str


@app.post("/login")
async def login(DB_login: Login):
    # 로그인 시 해당 이메일이 회원인지 확인
    if not check_existing_user(DB_login.member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email doesn't exist'"))
    data = {
        "sub": DB_login.member_email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    access_token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    post_slack_message(f"{DB_login.member_email}님이 로그인하셨습니다!")
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "member_email": DB_login.member_email,
    }


from inference.predict_signup import inference_signup


class Card(BaseModel):
    card_id_list: str
    space: str
    size: str
    family: str


@app.post("/card")
async def get_card_image(card: Card):  # 초기 보여주는 house
    card.card_id_list = list(
        map(int, filter(lambda x: x, card.card_id_list.strip("[]").split(",")))
    )
    card_id_result = card_vector.sampling_cards(card.card_id_list)
    return get_card_info(card_id_result)


class Signup(BaseModel):
    member_email: str


@app.post("/signup")
async def signup(db_signup: Signup, discription="회원가입 API입니다.") -> list:
    if check_existing_user(db_signup.member_email):
        return JSONResponse(status_code=400, content=dict(msg="Email already exist'"))
    else:
        return JSONResponse(status_code=200, content=dict(msg="Success'"))


class Image(BaseModel):
    member_email: str
    selected_card_id: str


@app.post("/member")
async def image(item: Image):
    # card_id -> house_id
    card_id_list = list(map(int, item.selected_card_id[1:-1].split(",")))
    house_id_list = []
    for card_id in card_id_list:
        house_id_list.append([col[0] for col in get_house_from_card(card_id)][0])

    return create_member(item.member_email, house_id_list)


"""
get : dict(json)를 받을 수 없음, {} 있을수도 없을수도
post : dict(json)를 받을 수 있음. {}로만 움직임.
"""


@app.get("/item/{item_id}")
async def detail(item_id: int):
    return get_item_info_all(item_id)[0]


class Mypage(BaseModel):
    member_email: str


@app.post("/mypage")
async def detail(mypage: Mypage):
    from service.item import get_item_info_prefer, get_item_prefer

    item_list = get_item_prefer(mypage.member_email)
    return get_item_info_prefer(item_list)


@app.get("/prefer/{member_email}/{item_id}")
async def is_prefer_item(member_email: str, item_id: int):
    """유저가 item에 좋아요를 눌렀는지 확인

    Args:
        member_email (str): 유저의 이메일
        item_id (int): 아이템 id
    """
    if len(check_is_prefer(member_email, item_id)) == 0:
        return False
    return True


@app.get("/insert-prefer/{member_email}/{item_id}")
async def insert_prefer_data(member_email: str, item_id: int):
    if len(check_is_prefer(member_email, item_id)) != 0:
        return "failure"
    return insert_member_prefer(member_email, item_id)


@app.delete("/delete-prefer/{member_email}/{item_id}")
async def delete_prefer_data(member_email: str, item_id: int):
    return delete_member_prefer(member_email, item_id)


class InferenceResult(BaseModel):
    member_email: str
    
@app.post('/insert-inference-result')
async def insert_inference_result(inference_result: InferenceResult, description="화원가입할 때 inference"):
    user_prefered_item = get_inference_input(inference_result.member_email)  # 모델에 넣을 input list(item_id_list)
    user_index  = check_user_index(inference_result.member_email)
    model_result = inference(user_prefered_item, MODEL, user_index)    # 모델 인퍼런스
    return create_inference(inference_result.member_email, model_result)


# 디비에 있는 좋아요를 꺼내와서 (아이템 아이디) inference input에


class UpdataInferenceResult(BaseModel):
    member_email: str


@app.post("/update-inference-result")
async def insert_inference_result(
    update_inference: UpdataInferenceResult, description="좋아요 반영 inference"
):
    member_prefer = get_item_prefer(update_inference.member_email)  #
    inference_result = [
        result[0] for result in get_inference_result(update_inference.member_email)
    ]
    user_prefered_item = get_inference_input(
        update_inference.member_email
    )  # 모델에 넣을 input list(item_id_list)

    update_list = member_prefer + user_prefered_item
    user_index  = check_user_index(update_inference.member_email)
    model_result = inference(update_list, MODEL, user_index)
    delete_inference(update_inference.member_email)
    difference = set(model_result) - set(inference_result)
    intersection = set(model_result) & set(inference_result)
    create_inference(update_inference.member_email, model_result),
    if difference:
        return {"new_item": get_item(difference), "inter": get_item(intersection)}
    else:
        return "Already Update"


@app.get("/cluster/{item_id}")
async def get_cluster(item_id: int):
    cluster_id = [row[0] for row in get_item_cluster(item_id)]
    print(cluster_id)
    if len(cluster_id) != 0:
        clusters = [cluster[0] for cluster in get_clusters(cluster_id, item_id)]
        clusters = random.sample(clusters, k=min(8, len(clusters)))
        return clusters
    else:
        return []


@app.get("/series/{item_id}")
async def get_same_series(item_id):
    item_info = get_item_info_all(int(item_id))
    category = item_info[0][1]
    seller = item_info[0][6]
    series_items = [col[0] for col in get_same_series_item(item_id, category, seller)]
    series_items = random.sample(series_items, k=min(8, len(series_items)))
    return series_items


@app.get("/popular/{item_id}")
async def get_same_series(item_id):
    item_info = get_item_info_all(int(item_id))
    category = item_info[0][1]
    popular_items = [col[0] for col in get_popular_item(item_id, category)]
    popular_items = random.sample(popular_items, k=min(8, len(popular_items)))
    return popular_items
