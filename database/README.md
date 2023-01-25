# db manager

## database 폴더 구조

database 디렉토리의 구조는 다음과 같습니다.

```
database
|- data
|   |- *.csv
|- sql
|   |- *.sql
|- db_manager.py
|- README.md
|- requirements.txt
```

DB 접속에 필요한 정보는 경로 최상단 secrets.yaml에 있습니다.  
data에는 DB에 넣을 csv 파일과 DB에서 나온 csv 파일이 있습니다. (TBD)  
sql에는 미리 지정된 sql 명령어 파일들이 들어있습니다.  
db_manager.py는 메인 실행 파일입니다.  

## db manager 사용 방법

1. requirements.txt를 설치합니다.  
2. 경로 최상단 secrets.yaml 파일의 DB 부분을 설정합니다. 해당 부분의 형식은 다음을 따릅니다.
```
DB:  
    user: root  
    password : [db password]  
    host : [db ip]  
    port : 3306  
    database : [database name]  
```  
3. ```python db_manager.py --command [create | delete]``` 명령어를 실행합니다.  

## 지정된 sql 명령어 모음 

명령어는 현재 다음 두가지가 있으며, 추후 업데이트 예정입니다.  

1. 데이터베이스 및 테이블 생성  
```python db_manager.py --command create```  
  
2. 테이블 삭제  
```python db_manager.py --command delete```  