# from main import fake_users
from db.models import Item
from db.models import House, Member
from sqlalchemy.orm import Session

from db.db_connect import Database
from db.models import House
from sqlalchemy import select

database = Database()

def check_existing_house(house_id: int):
    with database.session_maker() as session:
        stmt = select(House).where(House.house_id == house_id) # Statement -> DB Query를 의미
        # data를 어떤 형태(dict, list, ...)로 처리할까..
        return session.execute(stmt).first()

def check_existing_user(member_email: str) -> bool:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email == member_email) # Statement -> DB Query를 의미
        return True if session.execute(stmt).fetchall() else False
    

def save_db(member_email:str, card_info:list):
    with database.session_maker() as session:
        for i in card_info:
            stmt = Member(member_email=member_email, house_id=i)
            session.add(stmt)
            session.commit()

        return "Update complete"
    
