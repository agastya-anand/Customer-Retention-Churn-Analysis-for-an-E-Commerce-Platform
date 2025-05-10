# data_cleaning.py

# Import Libraries
import pandas as pd
import os

# Ensure the working directory is correct
print("Current Directory:", os.getcwd())

# Define Data Paths
customer_path = "data/customers.csv"
transaction_path = "data/transactions.csv"
engagement_path = "data/engagements.csv"

# Load Data
customers = pd.read_csv(customer_path)
transactions = pd.read_csv(transaction_path)
engagements = pd.read_csv(engagement_path)

# Display First Few Rows to Confirm Data Load
print("\nCustomers Data:\n", customers.head())
print("\nTransactions Data:\n", transactions.head())
print("\nEngagements Data:\n", engagements.head())

# Check for Missing Values in Each Dataset
print("\nMissing Values in Customers Data:\n", customers.isnull().sum())
print("\nMissing Values in Transactions Data:\n", transactions.isnull().sum())
print("\nMissing Values in Engagements Data:\n", engagements.isnull().sum())

# Handle Missing Values

# Customers Data
# Fill missing Membership values with 'Standard'
customers["Membership"].fillna("Standard", inplace=True)

# Transactions Data
# Fill missing PaymentMethod with 'Unknown'
transactions["PaymentMethod"].fillna("Unknown", inplace=True)

# Engagements Data
# Drop rows with missing values
engagements.dropna(inplace=True)

# Confirm the changes
print("\nMissing Values After Cleaning - Customers:\n", customers.isnull().sum())
print("\nMissing Values After Cleaning - Transactions:\n", transactions.isnull().sum())
print("\nMissing Values After Cleaning - Engagements:\n", engagements.isnull().sum())

# Check for Duplicates
print("\nDuplicate Rows in Customers:", customers.duplicated().sum())
print("Duplicate Rows in Transactions:", transactions.duplicated().sum())
print("Duplicate Rows in Engagements:", engagements.duplicated().sum())

# Remove Duplicates
customers.drop_duplicates(inplace=True)
transactions.drop_duplicates(inplace=True)
engagements.drop_duplicates(inplace=True)

# Confirm Removal
print("\nDuplicates After Removal - Customers:", customers.duplicated().sum())
print("Duplicates After Removal - Transactions:", transactions.duplicated().sum())
print("Duplicates After Removal - Engagements:", engagements.duplicated().sum())

# Convert Date Columns to Datetime
customers["JoinDate"] = pd.to_datetime(customers["JoinDate"])
transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
engagements["Date"] = pd.to_datetime(engagements["Date"])

# Confirm Data Types
print("\nData Types After Conversion - Customers:\n", customers.dtypes)
print("\nData Types After Conversion - Transactions:\n", transactions.dtypes)
print("\nData Types After Conversion - Engagements:\n", engagements.dtypes)

# Standardize Column Names
customers.columns = customers.columns.str.lower()
transactions.columns = transactions.columns.str.lower()
engagements.columns = engagements.columns.str.lower()

# Confirm the changes
print("\nColumn Names - Customers:\n", customers.columns)
print("\nColumn Names - Transactions:\n", transactions.columns)
print("\nColumn Names - Engagements:\n", engagements.columns)

# Create a 'cleaned_data' folder if it doesn't exist
if not os.path.exists("cleaned_data"):
    os.makedirs("cleaned_data")

# Save Cleaned Data
customers.to_csv("cleaned_data/customers_cleaned.csv", index=False)
transactions.to_csv("cleaned_data/transactions_cleaned.csv", index=False)
engagements.to_csv("cleaned_data/engagements_cleaned.csv", index=False)

print("\nData Cleaning Complete. Cleaned files saved to 'cleaned_data' folder.")

# Verify Cleaned Data
cleaned_customers = pd.read_csv("cleaned_data/customers_cleaned.csv")
print("\nFirst 5 Rows of Cleaned Customers Data:\n", cleaned_customers.head())