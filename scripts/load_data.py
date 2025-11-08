import sys
import os
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import sqlalchemy
from src.config.settings import Settings

s = Settings()
engine = sqlalchemy.create_engine(s.MYSQL_URI)

csv_map = {
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}

for table, file in csv_map.items():
    path = os.path.join("data", file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.to_sql(table, engine, if_exists="replace", index=False)
        print(f"Loaded {table} – {len(df)} rows")