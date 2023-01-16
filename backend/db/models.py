#TODO : DB쪽에서 작성해주실수 있나요?

# 1. DB를 띄우면서 테이블 생성
# 2. api 서버 동작할 때 테이블 생성

from db.db_connect import Base
from sqlalchemy import Column, String, Float, Integer

class User(Base):
    # TODO: Table 스키마 협의 되면 붙이기!
    # 조심할 것은 테이블의 column이 정확하게 일치해야함

    user_id = Column(String)
    item = Column(Integer)
    
class Item(Base):
    
    col1 = Column(String)
    col2 = Column(String)
    col_3 = Column(Integer)

