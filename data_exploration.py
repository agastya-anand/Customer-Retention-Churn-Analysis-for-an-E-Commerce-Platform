# data_exploration.py

# Import Libraries
import pandas as pd
import os

# Check Current Working Directory
print("Current Directory:", os.getcwd())

# Define Data Paths
customer_path = "data/customers.csv"
transaction_path = "data/transactions.csv"
engagement_path = "data/engagements.csv"

# Load Data
customers = pd.read_csv(customer_path)
transactions = pd.read_csv(transaction_path)
engagements = pd.read_csv(engagement_path)

# Display First 5 Rows of Each Dataset
print("\nCustomers Data:\n", customers.head())
print("\nTransactions Data:\n", transactions.head())
print("\nEngagements Data:\n", engagements.head())

# Check Data Types and Info
print("\nCustomers Data Info:")
print(customers.info())

print("\nTransactions Data Info:")
print(transactions.info())

print("\nEngagements Data Info:")
print(engagements.info())

# Check for Missing Values
print("\nMissing Values in Customers Data:\n", customers.isnull().sum())
print("\nMissing Values in Transactions Data:\n", transactions.isnull().sum())
print("\nMissing Values in Engagements Data:\n", engagements.isnull().sum())

# Statistical Summary - Customers
print("\nStatistical Summary - Customers:\n", customers.describe())

# Statistical Summary - Transactions
print("\nStatistical Summary - Transactions:\n", transactions.describe())

# Statistical Summary - Engagements
print("\nStatistical Summary - Engagements:\n", engagements.describe())

# Unique Values in Customers
print("\nUnique Membership Types:\n", customers["Membership"].value_counts())

# Unique Payment Methods
print("\nUnique Payment Methods:\n", transactions["PaymentMethod"].value_counts())

# Unique Channels
print("\nUnique Channels:\n", engagements["Channel"].value_counts())

