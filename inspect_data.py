import pandas as pd

customers = pd.read_csv("data/customers.csv")
transactions = pd.read_csv("data/transactions.csv")
engagements = pd.read_csv("data/engagements.csv")

print("\nCustomer Data Sample:")
print(customers.head())

print("\nTransaction Data Sample:")
print(transactions.head())

print("\nEngagement Data Sample:")
print(engagements.head())