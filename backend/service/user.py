from main import fake_users

def get_user_info(user_id):
    if user_id in fake_users:
        return fake_users[user_id]
    else:
        return "No user in database"
    
def check_duplicated_id(user_id):
    if user_id in fake_users:
        return False 
    else:
        return True
    
def insert_user(user_id:str, selected_img:str):
    # 유저 아이디 있는지 체크
    if check_duplicated_id(user_id):
        # 있으면 중복 되었다고 반환
        return "User already exists."

    else:
        # 없으면 DB에 유저 정보 저장
        new_user = {user_id : selected_img}
        fake_users.update(new_user)
        return "OK update complete"
        