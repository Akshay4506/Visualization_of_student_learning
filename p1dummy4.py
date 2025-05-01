# --- Import Required Libraries ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

# --- Load Dataset Dynamically ---
file_path = input("Enter the path of the dataset: ")  # Ask user for dataset path

try:
    data = pd.read_csv(file_path)
    print("\nDataset Loaded Successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Take only the first 100 rows(changable)
data = data.head(100)

# --- Required Columns Check (Updated to Match Your Dataset) ---
required_columns = [
    'Student_ID', 'Study_Hours_per_Week', 'Midterm_Score', 'Final_Score',
    'Assignments_Avg', 'Quizzes_Avg', 'Projects_Score', 'Total_Score'
]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit()

# --- 1. Heatmap: Performance Correlations ---
plt.figure(figsize=(8, 6))
sns.heatmap(data[['Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Projects_Score', 'Total_Score']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap of Student Performance")
plt.show()

# --- 2. Clustering: Study Hours vs Total Score ---
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Study_Hours_per_Week', 'Total_Score']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Study_Hours_per_Week', y='Total_Score', hue='Cluster', data=data, palette='viridis', s=100)
plt.title("Clustering of Students Based on Study Habits")
plt.xlabel("Study Hours per Week")
plt.ylabel("Total Score")
plt.show()

# --- 3. Regression: Predicting Final Score based on Study Hours ---
X = data[['Study_Hours_per_Week']]
y = data['Final_Score']
regressor = LinearRegression()
regressor.fit(X, y)
data['Predicted_Final_Score'] = regressor.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='Study_Hours_per_Week', y='Final_Score', data=data, line_kws={'color': 'red'})
plt.title("Regression: Study Hours vs Final Score")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Score")
plt.show()

# --- 4. Performance Trends (Improved Readability) ---
plt.figure(figsize=(12, 6))  # Increase figure size

# Plotting different scores
plt.plot(data['Student_ID'], data['Midterm_Score'], label='Midterm Score', marker='o', linestyle='-', markersize=6)
plt.plot(data['Student_ID'], data['Final_Score'], label='Final Score', marker='s', linestyle='--', markersize=6)
plt.plot(data['Student_ID'], data['Total_Score'], label='Total Score', marker='^', linestyle='-.', markersize=6)

plt.xlabel("Student ID")
plt.ylabel("Score")
plt.title("Student Performance Trends")
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better visibility
plt.xticks(rotation=90)

# Display only every 5th Student ID to reduce clutter
plt.xticks(ticks=data['Student_ID'][::5], labels=data['Student_ID'][::5])

plt.show()


# --- 5. Performance Distributions ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['Final_Score'], bins=10, kde=True, color='blue')
plt.title("Final Score Distribution")
plt.xlabel("Final Score")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(data['Assignments_Avg'], color='orange')
plt.title("Assignment Score Trends")
plt.tight_layout()
plt.show()

# --- 6. Study Habits vs Performance (BoxPlot) ---
# Binning Study Hours into Ranges to Reduce Congestion(grouping continuous numerical values into discrete intervals (or "bins"))
bins = [0, 5, 10, 15, 20, 25, 30, 35]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35']
data['Study_Hours_Binned'] = pd.cut(data['Study_Hours_per_Week'], bins=bins, labels=labels)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Study_Hours_Binned', y='Final_Score', data=data, hue='Study_Hours_Binned', palette='coolwarm', legend=False)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.xlabel("Study Hours per Week (Binned)")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score (Boxplot)")

plt.show()

# --- 7. Association Rule Mining ---
optional_columns = ['Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level']
valid_columns = [col for col in optional_columns if col in data.columns]

if valid_columns:
    basket = data[valid_columns]
    basket = pd.get_dummies(basket, drop_first=True)  # Convert categorical data to one-hot encoding
    frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)

    if not rules.empty:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(x=rules['confidence'], y=rules['antecedents'].astype(str), color='green')
        plt.xlabel("Confidence")
        plt.ylabel("Subjects")
        plt.title("Confidence of Subject Association")
        plt.subplot(1, 2, 2)
        sns.barplot(x=rules['lift'], y=rules['antecedents'].astype(str), color='purple')
        plt.xlabel("Lift Value")
        plt.ylabel("Subjects")
        plt.title("Lift of Subject Association")
        plt.tight_layout()
        plt.show()
    else:
        print("No significant association rules found. Try adjusting min_support or min_threshold.")
else:
    print("Skipping Association Rule Mining: Required columns missing.")
