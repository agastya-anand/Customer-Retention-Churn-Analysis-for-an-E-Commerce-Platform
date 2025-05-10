import pandas as pd
import numpy as np
from faker import Faker
import os
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)

# Create data directory if not exists
os.makedirs("data", exist_ok=True)

# Parameters
NUM_CUSTOMERS = 5000
NUM_TRANSACTIONS = 50000
NUM_ENGAGEMENTS = 100000

# Customer Data
def generate_customers(num_customers):
    customers = []
    membership_types = ['Basic', 'Silver', 'Gold', 'Platinum']
    
    for _ in range(num_customers):
        customers.append({
            "CustomerID": fake.unique.random_int(min=1000, max=9999),
            "Name": fake.name(),
            "Age": np.random.randint(18, 70),
            "Gender": np.random.choice(['Male', 'Female', 'Non-Binary']),
            "Location": fake.city(),
            "JoinDate": fake.date_this_decade(),
            "Membership": np.random.choice(membership_types, p=[0.4, 0.3, 0.2, 0.1])
        })
    
    return pd.DataFrame(customers)

# Transaction Data
def generate_transactions(num_transactions, customer_ids):
    transactions = []
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'COD']
    order_statuses = ['Completed', 'Cancelled', 'Returned']
    
    for _ in range(num_transactions):
        transactions.append({
            "TransactionID": fake.unique.random_int(min=10000, max=99999),
            "CustomerID": np.random.choice(customer_ids),
            "TransactionDate": fake.date_between(start_date='-1y', end_date='today'),
            "Amount": round(np.random.uniform(10, 500), 2),
            "PaymentMethod": np.random.choice(payment_methods),
            "OrderStatus": np.random.choice(order_statuses, p=[0.8, 0.1, 0.1])
        })
    
    return pd.DataFrame(transactions)

# Engagement Data
def generate_engagements(num_engagements, customer_ids):
    engagements = []
    channels = ['Email', 'SMS', 'Website', 'Social Media']
    actions = ['Opened', 'Clicked', 'Viewed', 'Liked']
    
    for _ in range(num_engagements):
        engagements.append({
            "EngagementID": fake.unique.random_int(min=100000, max=999999),
            "CustomerID": np.random.choice(customer_ids),
            "Date": fake.date_between(start_date='-1y', end_date='today'),
            "Channel": np.random.choice(channels),
            "Action": np.random.choice(actions)
        })
    
    return pd.DataFrame(engagements)

# Generate Data
customers_df = generate_customers(NUM_CUSTOMERS)
transactions_df = generate_transactions(NUM_TRANSACTIONS, customers_df['CustomerID'].tolist())
engagements_df = generate_engagements(NUM_ENGAGEMENTS, customers_df['CustomerID'].tolist())

# Save to CSV
customers_df.to_csv("data/customers.csv", index=False)
transactions_df.to_csv("data/transactions.csv", index=False)
engagements_df.to_csv("data/engagements.csv", index=False)

print("Data generation completed and saved to 'data' folder.")