import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ecommerce_data():
    np.random.seed(42)  # for reproducibility

    # Parameters
    n_customers = 5000
    n_transactions = 15000

    # Generate customer IDs
    customer_ids = [f'CUST_{i:05d}' for i in range(1, n_customers+1)]

    # We want some heavy buyers and many one-time buyers.
    # Let's assign each customer a probability of making multiple transactions.
    # We'll use a Pareto-like distribution: 80% of customers make only 1 transaction,
    # 15% make 2-5 transactions, 5% make 6-20 transactions.
    # But we need exactly n_transactions total.

    # Instead, we can generate transaction counts per customer from a distribution
    # and then adjust to meet the total.

    # Let's generate transaction counts from a negative binomial distribution
    # to get overdispersed counts (many zeros, few high counts).
    # However, we want at least one transaction per customer? Not necessarily.
    # But the problem says 5000 unique customers and 15000 transactions, so each customer has at least one?
    # It says "roughly 15000 transactions spread over a 2-year period" and "5000 unique customers".
    # We'll assume each customer has at least one transaction.

    # We'll generate counts such that the sum is n_transactions and each >=1.
    # We can generate from a Dirichlet distribution and then multiply by n_transactions.

    # Generate random weights for each customer
    weights = np.random.dirichlet(np.ones(n_customers)*0.5)  # parameter <1 for sparsity
    # Then assign counts
    counts = np.round(weights * n_transactions).astype(int)
    # Ensure each customer has at least 1 transaction by adding 1 to those with 0 and subtracting from others
    # But let's do a simpler approach: generate counts from a distribution and then adjust.

    # Instead, let's use a lognormal distribution and then adjust to the total.
    lognorm_counts = np.random.lognormal(mean=3, sigma=1.5, size=n_customers)
    # Scale to desired total
    counts = np.round(n_transactions * lognorm_counts / lognorm_counts.sum()).astype(int)
    # Ensure minimum 1
    counts[counts < 1] = 1
    # Adjust to exactly n_transactions by adding/subtracting from random customers
    diff = n_transactions - counts.sum()
    if diff != 0:
        # Adjust by adding/subtracting 1 from random customers
        indices = np.random.choice(n_customers, size=abs(diff), replace=True)
        if diff > 0:
            counts[indices] += 1
        else:
            # Ensure we don't go below 1
            for idx in indices:
                if counts[idx] > 1:
                    counts[idx] -= 1
                else:
                    # If already 1, we skip and choose another? But we might not subtract enough.
                    # Let's just adjust by selecting from those with count>1
                    pass
        # Recalculate diff and adjust again if needed (simplified)
        diff = n_transactions - counts.sum()
        if diff != 0:
            # Force adjustment by adding to the first customer
            counts[0] += diff

    # Now generate transactions for each customer
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 1, 1)
    days_between = (end_date - start_date).days

    for cust_id, n in zip(customer_ids, counts):
        # Generate n random dates within the 2-year period
        random_days = np.random.randint(0, days_between, size=n)
        order_dates = [start_date + timedelta(days=int(d)) for d in random_days]
        # Sort dates for each customer (optional, but realistic)
        order_dates.sort()
        # Generate order values: let's assume a Pareto distribution for order values
        # Most orders are small, few are large.
        # We'll use a lognormal distribution for order values.
        order_values = np.random.lognormal(mean=3, sigma=1, size=n)  # mean ~20, sigma 1 -> reasonable
        # Ensure positive
        order_values = np.maximum(order_values, 1.0)

        for date, value in zip(order_dates, order_values):
            data.append({
                'customer_id': cust_id,
                'order_date': date.strftime('%Y-%m-%d'),
                'order_value': round(value, 2)
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    # Sort by order_date
    df = df.sort_values('order_date').reset_index(drop=True)

    return df

if __name__ == "__main__":
    print("Generating e-commerce data...")
    df = generate_ecommerce_data()
    print(f"Generated {len(df)} transactions for {df['customer_id'].nunique()} unique customers.")
    print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    print("Saving to raw_orders.csv...")
    df.to_csv('raw_orders.csv', index=False)
    print("Done.")