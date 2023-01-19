from db.db_connect import Database
if __name__ == "__main__":
    import uvicorn
    try:
        print("Create tables")
        Database.create_user()
        print("Tables are created")
    except Exception as e:
        print("Table is already exists")

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)