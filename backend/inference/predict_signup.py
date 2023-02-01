from db.models import Card
from db.db_connect import Database
from sqlalchemy import select
import random
from sqlalchemy import func
import torch.nn as nn

'''
Input : [유저가 선택한 카드번호 리스트, 유저가 입력한 회원정보]
Output : 유저가 선택한 카드id 리스트 (10개)
'''
database = Database()

class Signup_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
    # def __init__(self, card_list, member_info) -> None:
    #     self.card_list = card_list
    #     self.member_info = member_info

    def forward(self, card_id, space, size, family):
        
        with database.session_maker() as session:
            stmt = select(Card).where(Card.img_space =='거실' and Card.is_human==0).order_by(func.random()).limit(10)
            data = session.execute(stmt).fetchall()
        return [col.Card.card_id for col in data]
    
model = Signup_Model()
def inference_signup(card_id, space, size, family):
    return model(card_id, space, size, family)