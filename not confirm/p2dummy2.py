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

# Generate Sample Educational Data
np.random.seed(42)  # Ensuring reproducibility
data = pd.DataFrame({
    'Student_ID': range(1, 101),
    'Quiz_Score': np.random.randint(50, 100, 100),
    'Assignment_Score': np.random.randint(40, 100, 100),
    'Video_Watch_Time': np.random.randint(10, 300, 100),  # Minutes
    'Engagement': np.random.randint(1, 10, 100),  # Scale 1-10
    'Study_Time': np.random.randint(1, 10, 100)
})

# --- 1. Clustering Analysis (K-Means) ---
X = data[['Quiz_Score', 'Assignment_Score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

# Plot Clustering
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Quiz_Score', y='Assignment_Score', hue='Cluster', data=data, palette='Set1')
plt.title("Clustering Students Based on Performance")
plt.xlabel("Quiz Score")
plt.ylabel("Assignment Score")
plt.show(block=True)

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Educational Factors")
plt.show(block=True)

# --- 3. Exam Score Distribution & Assignment Score Trends ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(data['Quiz_Score'], bins=10, kde=True, color='blue')
plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.boxplot(data['Assignment_Score'], color='orange')
plt.title("Assignment Score Trends")

plt.tight_layout()
plt.show()

# --- 4. Study Time vs Exam Score (Boxplot & Regression) ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=data['Study_Time'], y=data['Quiz_Score'], palette='coolwarm')
plt.title("Study Time vs Exam Score (Boxplot)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

plt.subplot(1, 2, 2)
sns.regplot(x='Study_Time', y='Quiz_Score', data=data, line_kws={'color': 'red'})
plt.title("Study Time vs Exam Score (Regression)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()

# --- 5. Regression Analysis (Predicting Quiz Score Based on Video Watch Time) ---
X = data[['Video_Watch_Time']]
y = data['Quiz_Score']
regressor = LinearRegression()
regressor.fit(X, y)
data['Predicted_Score'] = regressor.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='Video_Watch_Time', y='Quiz_Score', data=data, line_kws={'color': 'red'})
plt.title("Linear Regression: Video Watch Time vs Quiz Score")
plt.xlabel("Video Watch Time (minutes)")
plt.ylabel("Quiz Score")
plt.show()

# --- 6. Association Rule Mining (For Administrators) ---
basket = pd.DataFrame({
    'Math': np.random.choice([1, 0], 100),
    'Science': np.random.choice([1, 0], 100),
    'English': np.random.choice([1, 0], 100)
})

basket = basket.astype(bool)
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
    print("No significant association rules found. Try lowering min_support or min_threshold.")

# --- Final Summary ---
print("\n📢 Insights for Students, Teachers, and Administrators:")
print("🔹 Students: Check exam & assignment score trends to improve performance.")
print("🔹 Teachers: Understand study habits and their impact on scores.")
print("🔹 Administrators: Analyze subject associations for better curriculum design.")
