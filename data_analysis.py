import pandas as pd

# Load cleaned data
customers = pd.read_csv("cleaned_data/customers_cleaned.csv")
transactions = pd.read_csv("cleaned_data/transactions_cleaned.csv")
engagements = pd.read_csv("cleaned_data/engagements_cleaned.csv")

# Data Exploration: Customers Data
print("\nSummary Statistics - Customers Data:\n", customers.describe())

# Distribution of Customers by Membership
print("\nCustomer Distribution by Membership:\n", customers["membership"].value_counts())

# Distribution of Age
print("\nAge Distribution:\n", customers["age"].describe())

# Unique Locations
print("\nUnique Locations:\n", customers["location"].nunique())

# Check for Missing Values
print("\nMissing Values in Customers Data:\n", customers.isnull().sum())

# Data Exploration: Transactions Data
print("\nSummary Statistics - Transactions Data:\n", transactions.describe())

# Distribution of Payment Methods
print("\nPayment Method Distribution:\n", transactions["paymentmethod"].value_counts())

# Distribution of Order Status
print("\nOrder Status Distribution:\n", transactions["orderstatus"].value_counts())

# Data Exploration: Engagements Data
print("\nSummary Statistics - Engagements Data:\n", engagements.describe())

# Engagements by Channel
print("\nEngagements by Channel:\n", engagements["channel"].value_counts())

# Engagements by Action
print("\nEngagements by Action:\n", engagements["action"].value_counts())

# Check for Missing Values
print("\nMissing Values in Engagements Data:\n", engagements.isnull().sum())

# Data Analysis Example: Total Amount Spent by Each Customer
total_spent = transactions.groupby("customerid")["amount"].sum()
print("\nTotal Amount Spent by Each Customer:\n", total_spent)

# Data Analysis Example: Average Age of Customers by Membership
average_age_by_membership = customers.groupby("membership")["age"].mean()
print("\nAverage Age of Customers by Membership:\n", average_age_by_membership)

# Calculate Customer Lifetime Value (CLV)
clv = transactions.groupby("customerid")["amount"].sum()
clv = clv.sort_values(ascending=False)

print("\nTop 10 Customers by CLV:\n", clv.head(10))

# Convert TransactionDate to datetime
transactions["transactiondate"] = pd.to_datetime(transactions["transactiondate"])

# Create a 'month' column to group by
transactions['month'] = transactions['transactiondate'].dt.to_period('M')

# Count unique customers per month
monthly_customers = transactions.groupby("month")["customerid"].nunique()

# Calculate Retention Rate (percentage change in the number of customers)
retention_rate = monthly_customers.pct_change().fillna(0) * 100

print("\nCustomer Retention Rate (Monthly Change in Customers):\n", retention_rate)

# Identify customers who made purchases in each month
monthly_customer_ids = transactions.groupby("month")["customerid"].unique()

# Churn rate calculation (customers who didn't make a purchase the next month)
churned_customers = []
for i in range(1, len(monthly_customer_ids)):
    # Ensure that the previous month has customers to avoid division by zero
    if len(monthly_customer_ids[i-1]) > 0:
        churned = set(monthly_customer_ids[i-1]) - set(monthly_customer_ids[i])
        churned_customers.append(len(churned) / len(monthly_customer_ids[i-1]))  # Corrected division
    else:
        churned_customers.append(0)  # If no customers in the previous month, assume no churn

# Output churn rate
churn_rate = churned_customers
print("\nChurn Rate per Month:\n", churn_rate)

# Recency: Number of days since the last purchase
last_purchase_date = transactions.groupby("customerid")["transactiondate"].max()
recency = (pd.to_datetime("today") - last_purchase_date).dt.days

# Frequency: Number of purchases by the customer
frequency = transactions.groupby("customerid")["transactionid"].count()

# Monetary: Total spend by the customer
monetary = transactions.groupby("customerid")["amount"].sum()

# Combine RFM metrics
rfm = pd.DataFrame({"recency": recency, "frequency": frequency, "monetary": monetary})

# Define Recency, Frequency, and Monetary segments
rfm["recency_segment"] = pd.qcut(rfm["recency"], 4, labels=["1", "2", "3", "4"])  # 1 = most recent
rfm["frequency_segment"] = pd.qcut(rfm["frequency"], 4, labels=["1", "2", "3", "4"])  # 1 = most frequent
rfm["monetary_segment"] = pd.qcut(rfm["monetary"], 4, labels=["1", "2", "3", "4"])  # 1 = highest spender

# Create the RFM score
rfm["RFM_score"] = rfm["recency_segment"].astype(str) + rfm["frequency_segment"].astype(str) + rfm["monetary_segment"].astype(str)

print("\nRFM Segmentation of Customers:\n", rfm.head())

# Segment customers based on Age
age_bins = [18, 25, 35, 45, 55, 65, 100]
age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
customers["age_group"] = pd.cut(customers["age"], bins=age_bins, labels=age_labels)

# Segment customers based on Location
location_segments = customers["location"].value_counts().head(5)  # Top 5 locations
print("\nTop 5 Locations:\n", location_segments)

# Segment customers based on Membership
membership_segments = customers["membership"].value_counts()
print("\nMembership Segmentation:\n", membership_segments)

# Combine all segmentation into one dataframe
customer_segments = customers[["customerid", "age_group", "location", "membership"]]

print("\nCustomer Demographic Segmentation:\n", customer_segments.head())

import matplotlib.pyplot as plt

# Group by month and sum the amount to observe trend
monthly_sales = transactions.groupby("month")["amount"].sum()

# Plot the sales trend over time
plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='line', title='Total Sales Trend Over Time', color='green')
plt.xlabel("Month")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=45)
plt.show()

# Merge customers and transactions to get age and amount together
merged_data = pd.merge(transactions, customers[["customerid", "age"]], on="customerid")

# Calculate the correlation between age and transaction amount
correlation = merged_data[["age", "amount"]].corr()
print("\nCorrelation between Age and Transaction Amount:\n", correlation)

# Outlier Detection
# Detecting Outliers using IQR (Interquartile Range):

# Calculate IQR for Amount column
Q1 = transactions['amount'].quantile(0.25)
Q3 = transactions['amount'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = transactions[(transactions['amount'] < lower_bound) | (transactions['amount'] > upper_bound)]
print("\nOutliers in Amount:\n", outliers)

# Visualizing Outliers Using Boxplot

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot to visualize outliers in the 'amount' column
plt.figure(figsize=(10, 6))
sns.boxplot(x=transactions['amount'])
plt.title('Boxplot of Transaction Amounts')
plt.show()

# Using Z-Score for Outlier Detection

from scipy.stats import zscore

# Calculate Z-scores for the 'amount' column
transactions['zscore'] = zscore(transactions['amount'])

# Identify outliers where Z-score is greater than 3 or less than -3
outliers_zscore = transactions[transactions['zscore'].abs() > 3]
print("\nOutliers based on Z-score:\n", outliers_zscore)

# Time Series Forecasting (ARIMA Model)
# Ensure the 'TransactionDate' column is in datetime format
transactions['transactiondate'] = pd.to_datetime(transactions['transactiondate'])

# Set 'TransactionDate' as the index
transactions.set_index('transactiondate', inplace=True)

# Resample the data by month and aggregate the 'amount' (sum of transactions for each month)
monthly_sales = transactions['amount'].resample('M').sum()

from statsmodels.tsa.arima.model import ARIMA

# Define the ARIMA model (p, d, q) - You can fine-tune these parameters
model = ARIMA(monthly_sales, order=(5, 1, 0))
model_fit = model.fit()

# Forecast future sales for the next 6 months
forecast = model_fit.forecast(steps=6)
print("\nForecasted Sales for the Next 6 Months:", forecast)

import matplotlib.pyplot as plt

# Plot the actual vs predicted data
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, label='Actual Sales')
plt.plot(forecast, label='Forecasted Sales', color='red')
plt.legend()
plt.title('Time Series Forecasting of Monthly Sales')
plt.show()

# Customer Segmentation (Advanced)
# K-means Clustering
from sklearn.cluster import KMeans

# Merge transactions data with customers to get 'amount' for each customer
merged_data = pd.merge(customers, transactions[['customerid', 'amount']], on='customerid', how='left')

# Now you can use the 'amount' and 'age' columns for clustering
X = merged_data[['age', 'amount']]  # Example features for clustering

# Apply KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3)
merged_data['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.scatter(merged_data['age'], merged_data['amount'], c=merged_data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Amount')
plt.show()

# Check the column data types to find which columns are non-numeric
print(customers.dtypes)

# Select only numeric columns for correlation
numeric_columns = customers.select_dtypes(include=['number']).columns
correlation_matrix = customers[numeric_columns].corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Plot Purchase Frequency
purchase_frequency = transactions['customerid'].value_counts().reset_index()
purchase_frequency.columns = ['customerid', 'purchase_frequency']

# Plot Average Transaction Amount by Customer
avg_purchase_amount = transactions.groupby('customerid')['amount'].mean().reset_index()
avg_purchase_amount.columns = ['customerid', 'avg_purchase_amount']

# Visualize Purchase Frequency
plt.figure(figsize=(10, 6))
sns.histplot(purchase_frequency['purchase_frequency'], kde=True)
plt.title('Purchase Frequency Distribution')
plt.show()

# Visualize Average Purchase Amount
plt.figure(figsize=(10, 6))
sns.histplot(avg_purchase_amount['avg_purchase_amount'], kde=True)
plt.title('Average Purchase Amount Distribution')
plt.show()

#Churn prediction model
# Assume customers who haven't made a purchase in the last 30 days are considered churned
churn_threshold = 30  # days
rfm['churned'] = rfm['recency'] > churn_threshold

# Check balance of churned vs retained
churned_vs_retained = rfm['churned'].value_counts()
print(churned_vs_retained)

from sklearn.model_selection import train_test_split

# Define feature columns
X = rfm[['recency', 'frequency', 'monetary']]  # Use RFM as features
y = rfm['churned']  # Target variable

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluate Logistic Regression
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_pred)

# Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred)

# Evaluate XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_pred)

# Print all evaluation metrics
print(f"Logistic Regression - AUC: {lr_auc}, Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}, F1: {lr_f1}")
print(f"Random Forest - AUC: {rf_auc}, Accuracy: {rf_accuracy}, Precision: {rf_precision}, Recall: {rf_recall}, F1: {rf_f1}")
print(f"XGBoost - AUC: {xgb_auc}, Accuracy: {xgb_accuracy}, Precision: {xgb_precision}, Recall: {xgb_recall}, F1: {xgb_f1}")
