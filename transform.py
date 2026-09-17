import pandas as pd
import sqlite3
from datetime import datetime

def build_rfm_features():
    # Load the raw data
    print("Loading raw_orders.csv...")
    df = pd.read_csv('raw_orders.csv')
    # Convert order_date to datetime
    df['order_date'] = pd.to_datetime(df['order_date'])

    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    # Load the DataFrame into SQLite
    df.to_sql('orders', conn, index=False, if_exists='replace')

    # SQL query to compute RFM
    query = """
    WITH max_date AS (
        SELECT MAX(order_date) AS max_order_date FROM orders
    ),
    customer_agg AS (
        SELECT
            customer_id,
            MAX(order_date) AS last_order_date,
            COUNT(*) AS frequency,
            SUM(order_value) AS monetary
        FROM orders
        GROUP BY customer_id
    )
    SELECT
        customer_id,
        CAST((JULIANDAY(md.max_order_date) - JULIANDAY(last_order_date)) AS INTEGER) AS recency,
        frequency,
        monetary
    FROM customer_agg, max_date md
    ORDER BY customer_id;
    """

    print("Executing SQL query to compute RFM...")
    rfm_df = pd.read_sql_query(query, conn)
    conn.close()

    # Save to CSV
    print("Saving RFM features to sql_rfm_features.csv...")
    rfm_df.to_csv('sql_rfm_features.csv', index=False)
    print(f"RFM features shape: {rfm_df.shape}")
    print("Done.")

if __name__ == "__main__":
    build_rfm_features()