import argparse
import os
import json
import pymysql
import yaml
from pymysql.constants import CLIENT


def main():
    
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", default="create", type=str, help="create or delete database")
    args = parser.parse_args()
    
    # get secrets
    SECRET_FILE = os.path.join('../secrets.yaml')
    with open(SECRET_FILE) as fp:
        secrets = yaml.load(fp, yaml.FullLoader)
    DB = secrets["DB"]
    
    # connect to database
    conn = pymysql.connect(host=DB['host'], user=DB['user'], password=DB['password'], port=DB['port'], charset='utf8', client_flag=CLIENT.MULTI_STATEMENTS)
    cur = conn.cursor()
    
    # select command
    if (args.command == "create"): # Create database
        sql = open("sql/create_table.sql").read()
    elif (args.command == "delete"): # Delete databse 
        sql = open("sql/delete_table.sql").read()
        
    # execute
    cur.execute(sql)
    conn.commit()
    conn.close()
    

if __name__ == "__main__":
    main()