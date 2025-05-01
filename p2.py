import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Generate Sample Educational Data
np.random.seed(42)  # Ensuring reproducibility
data = pd.DataFrame({
    'Student_ID': range(1, 101),
    'Quiz_Score': np.random.randint(50, 100, 100),
    'Assignment_Score': np.random.randint(40, 100, 100),
    'Video_Watch_Time': np.random.randint(10, 300, 100),  # Minutes
    'Engagement': np.random.randint(1, 10, 100)  # Scale 1-10
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
plt.show(block=True)  # Keeps the plot open in IDLE

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Educational Factors")
plt.show(block=True)

# --- 3. Performance Development Visualization ---
data = data.sort_values(by='Student_ID')
plt.figure(figsize=(10, 5))
plt.plot(data['Student_ID'], data['Quiz_Score'], label='Quiz Score', marker='o', linestyle='-')
plt.plot(data['Student_ID'], data['Assignment_Score'], label='Assignment Score', marker='s', linestyle='--')
plt.xlabel("Student ID")
plt.ylabel("Score")
plt.title("Student Performance Development")
plt.legend()
plt.grid(True)
plt.show(block=True)

# --- 4️⃣ Regression Analysis (Predicting Quiz Score Based on Video Watch Time) ---
if 'Video_Watch_Time' in data.columns and 'Quiz_Score' in data.columns:
    X = data[['Video_Watch_Time']]  # Predictor
    y = data['Quiz_Score']  # Target Variable

    # Train Linear Regression Model
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Generate Predictions
    data['Predicted_Score'] = regressor.predict(X)

    # Plot Regression Line
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Video_Watch_Time', y='Quiz_Score', data=data, line_kws={'color': 'red'})
    plt.title("Linear Regression: Video Watch Time vs Quiz Score")
    plt.xlabel("Video Watch Time (minutes)")
    plt.ylabel("Quiz Score")
    plt.show()

else:
    print("Required columns for regression analysis are missing in the dataset!")

# Generate Predictions
predicted_scores = model.predict(X)

# Plot Predictions
plt.figure(figsize=(8, 5))
plt.scatter(y, predicted_scores, alpha=0.7, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--', color='red')  # 45-degree reference line
plt.xlabel("Actual Quiz Score")
plt.ylabel("Predicted Quiz Score")
plt.title("Regression Analysis: Predicted vs Actual Scores")
plt.grid(True)
plt.show(block=True)
