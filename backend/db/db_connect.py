import os
import yaml

SERECT_FILE = os.path.join('data/secrets.yaml')
with open(SERECT_FILE) as fp:
    serects = yaml.load(fp, yaml.FullLoader)

DB = serects["DB"]

DB_URL = f"mysql+pymysql://{DB['user']}:{DB['password']}@{DB['host']}/{DB['database']}"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Database:
    def __init__(self, db_url: str = DB_URL) -> None:
        self.engine = create_engine(db_url, pool_recycle = 500)
        self.session_maker = sessionmaker(self.engine)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        return self.session

    def connection(self):
        conn = self.engine.connect()
        return conn
    
    def create_user(self):
        # TODO: API 서버가 띄워질 때 테이블을 만들어야 하는 경우 해당 함수를 활용
    
        # print("Create tables")
        # Base.metadata.create_all()
        # print("Tables are created")
        ...
        
    # def get_db(self):
    #     db = self.session
    #     try:
    #         yield db
    #     finally:
    #         db.close()