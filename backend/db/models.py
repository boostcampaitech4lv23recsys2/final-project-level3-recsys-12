# 1. DB를 띄우면서 테이블 생성
# 2. api 서버 동작할 때 테이블 생성

from db.db_connect import Base
from sqlalchemy import Column, String, Integer, ForeignKey, BINARY
from sqlalchemy.orm import relationship


# 집들이
class House(Base):

    __tablename__ = "house" # MySQL DB Table 이름
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    space = Column(String(100), nullable=False) # 공간
    size = Column(String(100), nullable=True) # 평수
    work = Column(String(100), nullable=False) # 작업
    category = Column(String(100), nullable=False) # 분야
    family = Column(String(100), nullable=True) # 가족형태
    region = Column(String(100), nullable=True) # 지역
    style = Column(String(100), nullable=True) # 스타일
    detail = Column(String(100), nullable=True) # 세부공사
    
    # # Primary Key인 house_id를 house_item Table의 Foreign Key인 house_id와 연결
    # house_item = relationship("HouseItem", back_populates="house")
    # # Primary Key인 house_id를 house_color Table의 Foreign Key인 house_id와 연결
    # house_color = relationship("HouseColor", back_populates="house")
    # # Primary Key인 house_id를 member Table의 Foreign Key인 house_id와 연결
    # member = relationship("Member", back_populates="house")
    # # Primary Key인 house_id를 house_card Table의 Foreign Key인 house_id와 연결
    # house_card = relationship("HouseCard", back_populates="house")


# 가구
class Item(Base):
    
    __tablename__ = "item" # MySQL DB Table 이름
    item_id = Column(Integer, nullable=False, primary_key=True) # 가구 id: PK
    category = Column(String(100), nullable=True) # 카테고리
    rating = Column(String(100), nullable=True) # 평점
    review = Column(String(255), nullable=True) # 후기
    title = Column(String(100), nullable=False) # 제목
    seller = Column(String(100), nullable=False) # 판매자
    image = Column(String(255), nullable=False) # 사진
    price = Column(String(100), nullable=True) # 가격
    discount_rate = Column(String(100), nullable=True) # 할인률
    available_product = Column(String(100), nullable=True) # 판매여부
    predict_price = Column(String(100), nullable=True) # 예상가격
    
    # # Primary Key인 item_id를 house_item Table의 Foreign Key인 item_id와 연결
    # house_item = relationship("HouseItem", back_populates="item")


# 집들이_가구
class HouseItem(Base):
    
    __tablename__ = "house_item" # MySQL DB Table 이름
    # house_id = Column(Integer, ForeignKey("house.house_id")) # 집들이 id: FK
    # item_id = Column(Integer, ForeignKey("item.item_id")) # 가구 id: FK
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    item_id = Column(Integer, nullable=False, primary_key=True) # 가구 id: PK
    
    # # Foreign Key인 house_id와 item_id를 
    # # house Table의 Primary Key인 house_id와 
    # # item Table의 Primary Key인 item_id와 연결
    # house = relationship("House", back_populates="house_item")
    # item = relationship("Item", back_populates="house_item")
    
    
# 집들이 색상
class HouseColor(Base):
    
    __tablename__ = "house_color" # MySQL DB Table 이름
    # house_id = Column(Integer, ForeignKey("house.house_id")) # 집들이 id: FK
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    color_0 = Column(BINARY, nullable=False, default=0) # 검정색
    color_1 = Column(BINARY, nullable=False, default=0) # 하얀색
    color_2 = Column(BINARY, nullable=False, default=0) # 회색
    color_3 = Column(BINARY, nullable=False, default=0) # 베이지색
    color_4 = Column(BINARY, nullable=False, default=0) # 황토색
    color_5 = Column(BINARY, nullable=False, default=0) # 갈색
    color_6 = Column(BINARY, nullable=False, default=0) # 빨간색
    color_7 = Column(BINARY, nullable=False, default=0) # 분홍색
    color_8 = Column(BINARY, nullable=False, default=0) # 노란색
    color_9 = Column(BINARY, nullable=False, default=0) # 연두색
    color_10 = Column(BINARY, nullable=False, default=0) # 하늘색
    color_11 = Column(BINARY, nullable=False, default=0) # 파란색
    color_12 = Column(BINARY, nullable=False, default=0) # 남색
    
    # # Foreign Key인 house_id를 house Table의 Primary Key인 house_id와 연결
    # house = relationship("House", back_populates="house_color")
    
    
# 회원정보
class Member(Base):
    
    __tablename__ = "member" # MySQL DB Table 이름
    # house_id = Column(Integer, ForeignKey("house.house_id")) # 집들이 id: FK
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    member_email = Column(String(255), nullable=False) # 회원 email
    
    # # Foreign Key인 house_id를 house Table의 Primary Key인 house_id와 연결
    # house = relationship("House", back_populates="member")
    
    
# 카드
class Card(Base):
    
    __tablename__ = "card"
    card_id = Column(Integer, nullable=False, primary_key=True) # 카드 id: PK
    img_src = Column(String(255), nullable=True) # 사진 주소
    img_space = Column(String(100), nullable=True) # 공간
    
    # # Primary Key인 card_id를 house_card Table의 Foreign Key인 card_id와 연결
    # house_card = relationship("HouseCard", back_populates="card")
    
    
# 집들이_카드
class HouseCard(Base):
    
    __tablename__ = "house_card"
    # house_id = Column(Integer, ForeignKey("house.house_id")) # 집들이 id: FK
    # card_id = Column(Integer, ForeignKey("card.card_id")) # 카드 id: FK
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    card_id = Column(Integer, nullable=False, primary_key=True) # 카드 id: PK
    
    # # Foreign Key인 house_id와 card_id를
    # # house Table의 Primary Key인 house_id와
    # # card Table의 Primary Key인 card_id와 연결
    # house = relationship("House", back_populates="house_card")
    # card = relationship("Card", back_populates="house_card")