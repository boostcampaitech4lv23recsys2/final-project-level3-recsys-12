# PK-FK가 연결되어있는 것이 구현 난이도가 올라가는것 같아서 일단은 배제

# 1. DB를 띄우면서 테이블 생성
# 2. api 서버 동작할 때 테이블 생성

from db.db_connect import Base
from sqlalchemy import Column, String, Integer


# 집들이
class House(Base):

    __tablename__ = "house" # MySQL DB Table 이름
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    space = Column(String(100), nullable=True) # 공간
    size = Column(String(100), nullable=True) # 평수
    work = Column(String(100), nullable=True) # 작업
    category = Column(String(100), nullable=True) # 분야
    family = Column(String(100), nullable=True) # 가족형태
    region = Column(String(100), nullable=True) # 지역
    style = Column(String(100), nullable=True) # 스타일
    duration = Column(String(100), nullable=True) # 기간
    budget = Column(String(100), nullable=True) # 예산
    detail = Column(String(255), nullable=True) # 세부공사
    prefer = Column(Integer, nullable=True) # 선호
    scrab = Column(Integer, nullable=True) # 스크랩
    comment = Column(Integer, nullable=True) # 코멘트
    views = Column(Integer, nullable=True) # 조회수
    card_space = Column(String(100), nullable=True) # 카드 공간
    card_img_url = Column(String(255), nullable=True) # 카드주소


# 가구
class Item(Base):
    
    __tablename__ = "item" # MySQL DB Table 이름
    item_id = Column(Integer, nullable=False, primary_key=True) # 가구 id: PK
    category = Column(String(100), nullable=True) # 카테고리
    rating = Column(String(100), nullable=True) # 평점
    review = Column(String(255), nullable=True) # 후기
    price = Column(String(100), nullable=True) # 가격
    title = Column(String(100), nullable=True) # 제목
    seller = Column(String(100), nullable=True) # 판매자
    discount_rate = Column(String(100), nullable=True) # 할인률
    image = Column(String(255), nullable=True) # 사진
    available_product = Column(String(100), nullable=True) # 판매여부
    predict_price = Column(String(100), nullable=True) # 예상가격
    
    
# 회원정보
class Member(Base):
    
    __tablename__ = "member" # MySQL DB Table 이름
    member_email = Column(String(255), nullable=False, primary_key=True) # 회원 email: PK
    house_id = Column(Integer, nullable=True) # 집들이 id


# house, item interaction
class HouseItem(Base):
    
    __tablename__ = "house_item" # MySQL DB Table 이름
    house_id = Column(Integer, nullable=False, primary_key=True) # 회원 email: PK
    item_id = Column(Integer, nullable=True) # 집들이 id