DB_URL = ...

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Database:
    def __init__(self, db_url: str = DB_URL) -> None:
        self.engine = create_engine(db_url)
        self.session_maker = sessionmaker(self.engine)

    def create_tables(self):
        # TODO: API 서버가 띄워질 때 테이블을 만들어야 하는 경우 해당 함수를 활용

        # print("Create tables")
        # Base.metadata.create_all()
        # print("Tables are created")
        ...
