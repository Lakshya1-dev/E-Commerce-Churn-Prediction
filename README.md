# E-Commerce Customer Churn Prediction

## Business Problem
In e-commerce, understanding customer behavior is crucial for sustaining revenue and growth. Customer churn—when customers stop making purchases—directly impacts the bottom line. Traditional methods of identifying churn often rely on arbitrary time thresholds (e.g., no purchase in 90 days), which can misclassify seasonal or occasional buyers. This project presents a data-driven approach to predict customer churn using RFM (Recency, Frequency, Monetary) analysis and semi-supervised learning to handle the inherent ambiguity in labeling churn status.

## RFM SQL Architecture
The project leverages SQL for efficient feature engineering, demonstrating how complex aggregations can be performed directly in the database layer:

1. **Data Extraction**: Simulated e-commerce transaction data (`raw_orders.csv`) containing:
   - `customer_id`: Unique identifier
   - `order_date`: Timestamp of purchase
   - `order_value`: Transaction amount

2. **Feature Engineering with SQL**: Using SQLite, we compute RFM metrics for each customer:
   - **Recency**: Days since the customer's last order relative to the most recent order in the dataset
   - **Frequency**: Total number of orders placed by the customer
   - **Monetary**: Total sum of all order values

   The SQL query uses common table expressions (CTEs) to first find the maximum order date, then aggregate per-customer statistics, and finally calculate recency as a Julian day difference.

3. **Advantages of SQL-based RFM**:
   - Scalability: Can handle large datasets efficiently
   - Reusability: The same query can be integrated into production ETL pipelines
   - Transparency: Clear, auditable logic for feature generation

## Why Semi-Supervised Learning?
E-commerce churn prediction faces a unique challenge: 
- Clearly active customers (recent purchases)
- Clearly churned customers (no purchase for over a year)
- A large "gray area" of customers with intermediate recency values (e.g., 31-365 days) whose status is ambiguous

Traditional supervised learning requires fully labeled data, which we don't have for the gray area. Semi-supervised learning addresses this by:
- Using the small set of clearly labeled points (active and churned) as seeds
- Propagating labels to unlabeled points based on feature similarity (using LabelSpreading with a KNN kernel)
- Capturing the underlying manifold structure of the RFM space

This approach is particularly suitable because:
- It leverages the abundance of unlabeled data
- It assumes that similar customers (in RFM space) are likely to share the same churn status
- It provides a principled way to incorporate business knowledge (the 30/365 day thresholds) while letting the data drive predictions in uncertain regions

## Files in this Project
- `requirements.txt`: Python dependencies
- `extract.py`: Generates realistic e-commerce transaction data
- `transform.py`: Uses SQL to compute RFM features from raw transactions
- `analyze.py`: Applies LabelSpreading for churn prediction and visualizes results
- `raw_orders.csv`: Generated transaction data (created by extract.py)
- `sql_rfm_features.csv`: RFM matrix for each customer (created by transform.py)
- `rfm_with_predictions.csv`: RFM features with predicted churn labels (created by analyze.py)
- `churn_scatter.png`: Visualization of churn predictions (Recency vs Monetary)

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Generate data: `python extract.py`
3. Compute RFM features: `python transform.py`
4. Predict churn and visualize: `python analyze.py`

## Results
The scatter plot (`churn_scatter.png`) shows customers colored by predicted churn status:
- Green: Predicted as Active
- Red: Predicted as Churned

The model successfully identifies clear churned customers (high recency, varying monetary) and active customers (low recency), while making informed predictions for the gray area based on similarity to labeled points.

## Notes
- The data generation script creates a realistic long-tail distribution of customer purchasing behavior
- SQL query is designed for SQLite but can be adapted to other SQL dialects
- LabelSpreading parameters (like `n_neighbors`) can be tuned based on business context
- This project demonstrates a complete end-to-end pipeline from data generation to actionable insights