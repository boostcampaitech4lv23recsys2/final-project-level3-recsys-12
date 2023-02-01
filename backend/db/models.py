# PK-FK가 연결되어있는 것이 구현 난이도가 올라가는것 같아서 일단은 배제

# 1. DB를 띄우면서 테이블 생성
# 2. api 서버 동작할 때 테이블 생성

from db.db_connect import Base
from sqlalchemy import Column, String, Integer, BINARY


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
    idx = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    member_email = Column(String(255), nullable=False) # 회원 email
    house_id = Column(Integer, nullable=False) # 집들이 id
    

# 회원선호
class MemberPrefer(Base):
    
    __tablename__ = "member_prefer" # MySQL DB Table 이름
    idx = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    member_email = Column(String(255), nullable=False) # 회원 email
    item_id = Column(Integer, nullable=False) # 가구 id
    
    
# 집들이_가구 interaction
class HouseItem(Base):
    
    __tablename__ = "house_item" # MySQL DB Table 이름
    idx = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    house_id = Column(Integer, nullable=False) # 집들이 id
    item_id = Column(Integer, nullable=False) # 가구 id


# 인퍼런스 결과 저장
class InferenceResult(Base):
    
    __tablename__ = "inference_result" # MySQL DB Table 이름
    idx = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    member_email = Column(String(255), nullable=False) # 집들이 id
    item_id = Column(Integer, nullable=False) # 가구 id
    
    
# 클러스터_가구 match
class ClusterItem(Base):
    
    __tablename__ = "cluster_item" # MySQL DB Table 이름
    idx = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    cluster_id = Column(Integer, nullable=False) # 클러스터 id
    item_id = Column(Integer, nullable=False) # 가구 id

# 클러스터_가구 match
class Card(Base):
    
    __tablename__ = "card" # MySQL DB Table 이름
    card_id = Column(Integer, nullable=False, autoincrement=True, primary_key=True) # 인덱스 (사용 x)
    img_space = Column(String(100), nullable=True) # 카드 공간
    img_url = Column(String(255), nullable=True) # 카드주소
    house_id = Column(Integer, nullable=False)
    is_human = Column(BINARY, nullable=False)
    
# 카드
class Card(Base):
    
    __tablename__ = "card" # MySQL DB Table 이름
    card_id = Column(Integer, nullable=False, primary_key=True) # 카드 id
    img_space = Column(String(100), nullable=True) # 이미지공간
    img_url = Column(String(255), nullable=True) # 이미지주소
    house_id = Column(Integer, nullable=True) # 집들이 id
    is_human = Column(BINARY, nullable=False, default=0) # 사람여부
    
    
# 집들이 색상
class HouseColor(Base):
    
    __tablename__ = "house_color" # MySQL DB Table 이름
    house_id = Column(Integer, nullable=False, primary_key=True) # 집들이 id: PK
    main_0 = Column(BINARY, nullable=False, default=0) # 전체_검은색
    main_1 = Column(BINARY, nullable=False, default=0) # 전체_하얀색
    main_2 = Column(BINARY, nullable=False, default=0) # 전체_회색
    main_3 = Column(BINARY, nullable=False, default=0) # 전체_베이지색
    main_4 = Column(BINARY, nullable=False, default=0) # 전체_황토색
    main_5 = Column(BINARY, nullable=False, default=0) # 전체_갈색
    main_6 = Column(BINARY, nullable=False, default=0) # 전체_빨간색
    main_7 = Column(BINARY, nullable=False, default=0) # 전체_분홍색
    main_8 = Column(BINARY, nullable=False, default=0) # 전체_노란색
    main_9 = Column(BINARY, nullable=False, default=0) # 전체_연두색
    main_10 = Column(BINARY, nullable=False, default=0) # 전체_하늘색
    main_11 = Column(BINARY, nullable=False, default=0) # 전체_파란색
    main_12 = Column(BINARY, nullable=False, default=0) # 전체_남색
    wall_0 = Column(BINARY, nullable=False, default=0) # 벽_검은색
    wall_1 = Column(BINARY, nullable=False, default=0) # 벽_하얀색
    wall_2 = Column(BINARY, nullable=False, default=0) # 벽_회색
    wall_3 = Column(BINARY, nullable=False, default=0) # 벽_베이지색
    wall_4 = Column(BINARY, nullable=False, default=0) # 벽_황토색
    wall_5 = Column(BINARY, nullable=False, default=0) # 벽_갈색
    wall_6 = Column(BINARY, nullable=False, default=0) # 벽_빨간색
    wall_7 = Column(BINARY, nullable=False, default=0) # 벽_분홍색
    wall_8 = Column(BINARY, nullable=False, default=0) # 벽_노란색
    wall_9 = Column(BINARY, nullable=False, default=0) # 벽_연두색
    wall_10 = Column(BINARY, nullable=False, default=0) # 벽_하늘색
    wall_11 = Column(BINARY, nullable=False, default=0) # 벽_파란색
    wall_12 = Column(BINARY, nullable=False, default=0) # 벽_남색
    floor_0 = Column(BINARY, nullable=False, default=0) # 바닥_검은색
    floor_1 = Column(BINARY, nullable=False, default=0) # 바닥_하얀색
    floor_2 = Column(BINARY, nullable=False, default=0) # 바닥_회색
    floor_3 = Column(BINARY, nullable=False, default=0) # 바닥_베이지색
    floor_4 = Column(BINARY, nullable=False, default=0) # 바닥_황토색
    floor_5 = Column(BINARY, nullable=False, default=0) # 바닥_길색
    floor_6 = Column(BINARY, nullable=False, default=0) # 바닥_빨간색
    floor_7 = Column(BINARY, nullable=False, default=0) # 바닥_분홍색
    floor_8 = Column(BINARY, nullable=False, default=0) # 바닥_노란색
    floor_9 = Column(BINARY, nullable=False, default=0) # 바닥_연두색
    floor_10 = Column(BINARY, nullable=False, default=0) # 바닥_하늘색
    floor_11 = Column(BINARY, nullable=False, default=0) # 바닥_파란색
    floor_12 = Column(BINARY, nullable=False, default=0) # 바닥_남색
