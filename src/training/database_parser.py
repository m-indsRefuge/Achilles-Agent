# src/training/database_parser.py

import sqlite3
from typing import List, Dict
import pymongo


def load_sqlite_database(db_path: str, table: str) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    columns = [desc[0] for desc in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))
    conn.close()
    return results


def load_mongodb(uri: str, database: str, collection: str) -> List[Dict]:
    client = pymongo.MongoClient(uri)
    coll = client[database][collection]
    return list(coll.find({}, {"_id": 0}))
