import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

# Sample data for demonstration
# Creating a sample dataset with student data
np.random.seed(42)  # Ensure reproducibility
data = pd.DataFrame({
    'student_id': range(1, 21),
    'study_time': np.random.randint(1, 10, 20),
    'exam_score': np.random.randint(40, 100, 20),
    'assignment_score': np.random.randint(20, 100, 20)
})

# --------- 1. Heatmap (Visualizing student performance correlations) ---------
plt.figure(figsize=(8, 6))
sns.heatmap(data[['exam_score', 'assignment_score']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap of Student Performance")
plt.show(block=True)

# --------- 2. Clustering (Grouping students based on study time and scores) ---------
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['cluster'] = kmeans.fit_predict(data[['study_time', 'exam_score']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='study_time', y='exam_score', hue='cluster', data=data, palette='viridis', s=100)
plt.title("Clustering of Students Based on Study Habits")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")
plt.show()

# --------- 3. Regression (Predicting Exam Score based on Study Time) ---------
X = data[['study_time']]
y = data['exam_score']

regressor = LinearRegression()
regressor.fit(X, y)

data['predicted_score'] = regressor.predict(X)

plt.figure(figsize=(8, 6))
sns.regplot(x='study_time', y='exam_score', data=data, line_kws={'color': 'red'})
plt.title("Linear Regression: Study Time vs Exam Score")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")
plt.show()

# --------- 4. Performance Development Visualization ---------
data = data.sort_values(by='student_id')

plt.figure(figsize=(10, 5))
plt.plot(data['student_id'], data['exam_score'], label='Exam Score', marker='o', linestyle='-')
plt.plot(data['student_id'], data['assignment_score'], label='Assignment Score', marker='s', linestyle='--')

# Labels and title
plt.xlabel("Student ID")
plt.ylabel("Score")
plt.title("Student Performance Development")

# Display legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show(block=True)

# --------- 5. Association Rule Mining (Finding Learning Patterns) ---------
# Sample dataset for association rules
basket = pd.DataFrame({
    'Math': [1, 0, 1, 1, 0],
    'Science': [0, 1, 1, 0, 1],
    'English': [1, 1, 0, 1, 0]
})

# Convert integer data (0/1) to Boolean (True/False) for mlxtend compatibility
basket = basket.astype(bool)

# Adjust min_support to find more itemsets (0.2 means appears in at least 20% of rows)
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)


# Generate association rules with a lower threshold for more results
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)

# Display rules
if not rules.empty:
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("\nNo significant association rules found. Try lowering min_support or min_threshold.")

