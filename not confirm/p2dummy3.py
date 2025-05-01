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
file_path = input("Enter the path of the dataset: ")
data = pd.read_csv(file_path)

# Check for required columns
required_columns = {'Midterm_Score', 'Final_Score', 'Projects_Score', 'Study_Hours_per_Week', 'Attendance (%)', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'}
available_columns = set(data.columns)
missing_columns = required_columns - available_columns

if missing_columns:
    print(f"Warning: The following required columns are missing from the dataset: {missing_columns}")
else:
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
    plt.figure(figsize=(6, 4))
    sns.heatmap(data[list(required_columns)].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
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

    # --- 4. Study Hours vs Final Score (Regression) ---
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Study_Hours_per_Week', y='Final_Score', data=data, line_kws={'color': 'red'})
    plt.title("Study Hours vs Final Score (Regression)")
    plt.xlabel("Study Hours per Week")
    plt.ylabel("Final Score")
    plt.show()

    # --- 5. Regression Analysis (Predicting Final Score Based on Study Hours) ---
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

    # --- 6. Association Rule Mining ---
    categorical_columns = {'Department', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level'}
    available_categorical = list(categorical_columns & available_columns)

    if available_categorical:
        categorical_data = pd.get_dummies(data[available_categorical])
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
    else:
        print("Not enough categorical data for association rule mining.")
