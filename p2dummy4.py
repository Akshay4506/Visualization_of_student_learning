import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset from storage
file_path = input("Enter the path of the dataset: ")  # Ask user for dataset path

try:
    data = pd.read_csv(file_path)
    print("\nDataset Loaded Successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Take only the first 250 rows(changable)
data = data.head(250)

# --- 1. Clustering Analysis (K-Means) ---
X = data[['Midterm_Score', 'Final_Score', 'Projects_Score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

# Plot Clustering
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Midterm_Score', y='Final_Score', hue='Cluster', data=data, palette='Set1')
plt.title("Clustering Students Based on Performance")
plt.xlabel("Midterm Score")
plt.ylabel("Final Score")
plt.show()

# --- 2. Correlation Heatmap ---
required_columns = [
    'Midterm_Score', 'Final_Score', 'Projects_Score',
    'Study_Hours_per_Week', 'Attendance (%)', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
]
data_subset = data[required_columns]  # Ensure only relevant columns are used
plt.figure(figsize=(6, 4))
sns.heatmap(data_subset.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Academic & Well-being Factors")
plt.show()

# --- 3. Exam Score Distribution & Assignment Score Trends ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['Final_Score'], bins=10, kde=True, color='blue')
plt.title("Final Exam Score Distribution")
plt.xlabel("Final Score")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(y=data['Projects_Score'], color='orange')
plt.title("Projects Score Trends")

plt.tight_layout()
plt.show()

# --- 4. Regression Analysis (Predicting Final Score Based on Study Hours) ---
X = data[['Study_Hours_per_Week']]
y = data['Final_Score']
regressor = LinearRegression()
regressor.fit(X, y)
data['Predicted_Final_Score'] = regressor.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='Study_Hours_per_Week', y='Final_Score', data=data, line_kws={'color': 'red'})
plt.title("Linear Regression: Study Hours vs Final Score")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Score")
plt.show()

# --- 5. Association Rule Mining ---
categorical_data = data[['Department', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level']]
categorical_data = pd.get_dummies(categorical_data)
frequent_itemsets = apriori(categorical_data, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)

if not rules.empty:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=rules['confidence'], y=rules['antecedents'].astype(str), color='green')
    plt.xlabel("Confidence")
    plt.ylabel("Factors")
    plt.title("Confidence of Factor Association")

    plt.subplot(1, 2, 2)
    sns.barplot(x=rules['lift'], y=rules['antecedents'].astype(str), color='purple')
    plt.xlabel("Lift Value")
    plt.ylabel("Factors")
    plt.title("Lift of Factor Association")

    plt.tight_layout()
    plt.show()
else:
    print("No significant association rules found.")
