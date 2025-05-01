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
    'assignment_score': np.random.randint(20, 100, 20),
    'video_watch_time': np.random.randint(30, 300, 20)  # New Feature
})

# --------- 3. Student Performance Overview (For Students) ---------
plt.figure(figsize=(12, 5))

# Exam Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(data['exam_score'], bins=10, kde=True, color='blue')
plt.title("📚 Exam Score Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")

# Assignment Score Trends
plt.subplot(1, 2, 2)
sns.boxplot(data['assignment_score'], color='orange')
plt.title("📝 Assignment Score Trends")

plt.tight_layout()
plt.show()

# 📌 **Student Insights:**
# - Helps students see **how their exam scores compare** to others.
# - Highlights **common assignment score ranges**.

# --------- 4. Study Habits & Exam Performance (For Teachers) ---------
plt.figure(figsize=(12, 5))

# Boxplot: Study Time vs Exam Score
plt.subplot(1, 2, 1)
sns.boxplot(x=data['study_time'], y=data['exam_score'], palette='coolwarm')
plt.title("📖 Study Time vs Exam Score (Boxplot)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

# Scatterplot with Regression Line
plt.subplot(1, 2, 2)
sns.regplot(x='study_time', y='exam_score', data=data, line_kws={'color': 'red'})
plt.title("📊 Study Time vs Exam Score (Regression)")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

plt.tight_layout()
plt.show()

# 📌 **Teacher Insights:**
# - Helps understand **study time efficiency**.
# - Boxplot shows **distribution of study hours** across different exam scores.
# - Regression line shows **impact of study time on performance**.

# --------- 5. Video Learning & Performance Analysis ---------
plt.figure(figsize=(12, 5))

# Scatterplot: Video Watch Time vs Exam Score
plt.subplot(1, 2, 1)
sns.regplot(x='video_watch_time', y='exam_score', data=data, line_kws={'color': 'red'})
plt.title("📺 Video Watch Time vs Exam Score")
plt.xlabel("Video Watch Time (minutes)")
plt.ylabel("Exam Score")

# Correlation Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("📌 Correlation Heatmap")

plt.tight_layout()
plt.show()

# 📌 **Insights:**
# - Determines if **watching more video content** improves student scores.
# - Heatmap helps identify **strong relationships between features**.

# --------- 6. Clustering Analysis (For Administrators) ---------
X = data[['study_time', 'exam_score']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='study_time', y='exam_score', hue='Cluster', data=data, palette='Set1')
plt.title("📌 Clustering Students Based on Study Time & Exam Scores")
plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")

# Annotate Performance Zones
plt.text(2, 90, "🔹 High Performers", fontsize=10, color='black')
plt.text(5, 50, "🔹 Average Students", fontsize=10, color='black')
plt.text(8, 40, "🔹 Needs Improvement", fontsize=10, color='black')

plt.show()

# 📌 **Administrator Insights:**
# - Groups students into **low, medium, and high performance** clusters.
# - Helps administrators **design targeted learning interventions**.

# --------- 7. Association Rule Mining (For Administrators) ---------
# Sample subject dataset
basket = pd.DataFrame({
    'Math': [1, 0, 1, 1, 0],
    'Science': [0, 1, 1, 0, 1],
    'English': [1, 1, 0, 1, 0],
    'History': [0, 1, 1, 0, 1]
})

# Convert to boolean
basket = basket.astype(bool)

# Generate frequent itemsets
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
    plt.title("🔗 Confidence of Subject Association")

    # Lift Bar Chart
    plt.subplot(1, 2, 2)
    sns.barplot(x=rules['lift'], y=rules['antecedents'].astype(str), color='purple')
    plt.xlabel("Lift Value")
    plt.ylabel("Subjects")
    plt.title("📈 Lift of Subject Association")

    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ No significant association rules found. Try adjusting min_support.")

# 📌 **Administrator Insights:**
# - **Confidence**: Likelihood of taking one subject if another is taken.
# - **Lift > 1**: Indicates **strong relationships** between subjects.
# - Helps in **curriculum planning & subject combinations**.

# --------- 📢 Final Summary of Graphical Insights ---------
print("\n📢 Insights for Students, Teachers, and Administrators:")
print("🔹 **Students**: Check exam & assignment score trends to improve performance.")
print("🔹 **Teachers**: Understand study habits, video learning impact, and engagement.")
print("🔹 **Administrators**: Use clustering & association rules to optimize education strategies.")
