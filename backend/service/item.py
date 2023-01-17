from db.models import Item
from db.db_connect import Database
from sqlalchemy import select
from inference.predict import inference
from main import fake_users

database = Database()

def get_item(item_ids):
    
    # Read data
    item_infos = {}
    for item_id in item_ids:
        with database.session_maker() as session:
            stmt = select(Item).where(Item.id == item_id) # Statement -> DB Query를 의미
            item_info = session.execute(stmt)
            item_infos[item_id] = item_info
            
    return item_ids, item_infos
    