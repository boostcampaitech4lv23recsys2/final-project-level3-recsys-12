# db manager 사용법

우선은 간단하게만 적고 추후 리드미 업데이트 및 노션 기록 하겠습니다.   

database/config 폴더 생성 후, 그 안에 secrets.json 파일을 생성합니다.  
json 파일은 다음 형식을 따라야 합니다.  

{  
    "host": <ins>database ip</ins>,  
    "user": "root",  
    "password": <ins>database password</ins>,  
    "port": 3306  
}  

명령어는 현재 다음 두가지가 있으며, 추후 업데이트 예정입니다.  

1. 데이터베이스 및 테이블 생성  
python db_manager.py --command create  
  
2. 테이블 삭제  
python db_manager.py --command delete  