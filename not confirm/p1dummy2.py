# --- 1. Import Required Libraries ---
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

# --- 2. Sample Dataset ---
np.random.seed(42)  # Ensure reproducibility
data = pd.DataFrame({
    'student_id': range(1, 21),
    'study_time': np.random.randint(1, 10, 20),
    'exam_score': np.random.randint(40, 100, 20),
    'assignment_score': np.random.randint(20, 100, 20)
})

# --------- 1. Heatmap (Visualizing Student Performance Correlations) ---------
plt.figure(figsize=(8, 6))
sns.heatmap(data[['exam_score', 'assignment_score']].corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap of Student Performance")
plt.show(block=True)

# --------- 2. Clustering (Grouping Students Based on Study Time & Scores) ---------
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
data['cluster'] = kmeans.fit_predict(data[['study_time', 'exam_score']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x='study_time', y='exam_score', hue='cluster', data=data, palette='viridis', s=100)
plt.title("Clustering of Students Based on Study Habits")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")
plt.show()

# --------- 3. Regression (Predicting Exam Score Based on Study Time) ---------
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

# --------- 5. Student Performance Overview (For Students) ---------
plt.figure(figsize=(12, 5))

# Exam Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(data['exam_score'], bins=10, kde=True, color='blue')
plt.title("Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")

# Assignment Score Trends
plt.subplot(1, 2, 2)
sns.boxplot(data['assignment_score'], color='orange')
plt.title("Assignment Score Trends")

plt.tight_layout()
plt.show()

# 📌 **Student Insights:**
# - Helps students see **how their exam scores compare** to others.
# - Highlights **common assignment score ranges**.

# --------- 6. Study Habits & Exam Performance (For Teachers) ---------
plt.figure(figsize=(12, 5))

# Boxplot: Study Time vs Exam Score
plt.subplot(1, 2, 1)
sns.boxplot(x=data['study_time'], y=data['exam_score'], palette='coolwarm')
plt.title("Study Time vs Exam Score (Boxplot)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

# Scatterplot with Regression Line
plt.subplot(1, 2, 2)
sns.regplot(x='study_time', y='exam_score', data=data, line_kws={'color': 'red'})
plt.title("Study Time vs Exam Score (Regression)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()

# 📌 **Teacher Insights:**
# - Helps understand **study time efficiency**.
# - Boxplot shows **distribution of study hours** across different exam scores.
# - Regression line shows **impact of study time on performance**.

# --------- 7. Association Rule Mining (For Administrators) ---------
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

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.2)

# Plot Subject Association Rules
if not rules.empty:
    plt.figure(figsize=(12, 5))

    # Confidence Bar Chart
    plt.subplot(1, 2, 1)
    sns.barplot(x=rules['confidence'], y=rules['antecedents'].astype(str), color='green')
    plt.xlabel("Confidence")
    plt.ylabel("Subjects")
    plt.title("Confidence of Subject Association")

    # Lift Bar Chart
    plt.subplot(1, 2, 2)
    sns.barplot(x=rules['lift'], y=rules['antecedents'].astype(str), color='purple')
    plt.xlabel("Lift Value")
    plt.ylabel("Subjects")
    plt.title("Lift of Subject Association")

    plt.tight_layout()
    plt.show()
else:
    print("\nNo significant association rules found. Try lowering min_support or min_threshold.")

# 📌 **Administrator Insights:**
# - **Confidence**: Likelihood of taking one subject if another is taken.
# - **Lift > 1**: Strong relationships between subjects.
# - Helps in **curriculum planning & subject combinations**.

# --------- 📢 Final Summary of Graphical Insights ---------
print("\nInsights for Students, Teachers, and Administrators:")
print("**Students**: Check exam & assignment score trends to improve performance.")
print("**Teachers**: Understand student study habits and their impact on scores.")
print("**Administrators**: Analyze subject associations for better curriculum design.")
