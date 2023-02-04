# from main import fake_users
from db.db_connect import Database
from db.models import *
from db.models import House
from sqlalchemy import select
from sqlalchemy.orm import Session

database = Database()

"""
이메일로 가입했을 때 확인 코드를 받고 맞으면 가입 -> 임시 비밀번호
"""


def check_existing_house(house_id: int):
    with database.session_maker() as session:
        stmt = select(House).where(House.house_id == house_id)
        return session.execute(stmt).first()


def check_existing_user(member_email: str) -> bool:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email == member_email)
        return True if session.execute(stmt).fetchall() else False


def get_house_from_card(card_id):
    with database.session_maker() as session:
        stmt = select(Card.house_id).where(Card.card_id == card_id)
        return session.execute(stmt).fetchall()


def create_member(member_email: str, house_id_list: str):
    """_summary_

    Args:
        member_email (str): str
        card_info (dict): member_email : house_id

    Returns:
        _type_: message
    """

    with database.session_maker() as session:
        data = select(Member.cnt).order_by(Member.cnt.desc()).limit(1)
        count = session.execute(data).fetchall()
        if not count:
            last_count = 0
        else:
            last_count = count[0].cnt
        print(last_count)
        for i in house_id_list:
            stmt = Member(member_email=member_email, house_id=i, cnt=last_count + 1)
            session.add(stmt)
            session.commit()

        return "Update complete"


def get_users_email():
    with database.session_maker() as session:
        stmt = "select distinct member_email from member"
        return session.execute(stmt).fetchall()


def create_inference(member_email, inference_item_list):
    with database.session_maker() as session:
        for item_id in inference_item_list:
            stmt = InferenceResult(member_email=member_email, item_id=item_id)
            session.add(stmt)
            session.commit()

        return "success"

def check_user_index(member_email: str) -> int:
    with database.session_maker() as session:
        stmt = select(Member).where(Member.member_email == member_email)
        data = session.execute(stmt).fetchall()
        print(data) # garbage collector

        return ([col[0].cnt for col in data])[0]