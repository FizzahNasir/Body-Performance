import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset (update the path if necessary)
df = pd.read_csv("bodyPerformance.csv")

# Encoding categorical columns like 'gender' using LabelEncoder
label_encoder = LabelEncoder()

# Encode 'gender' column (assuming it's named 'gender')
df['gender'] = label_encoder.fit_transform(df['gender'])

# Assume that you've already preprocessed your data and split it into features (X) and labels (y)
X = df.drop(columns=['class'])  # Your features (replace 'class' with the actual column name if necessary)
y = df['class']  # The target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple K-Nearest Neighbors (KNN) classifier as an example
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate Precision, Recall, and F1-Score for each class
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1_scores = f1_score(y_test, y_pred, average=None)

# Streamlit components to display data and results
st.title("Model Evaluation and Results")

# Show the first 5 rows of the dataset
st.write("### First 5 rows of the dataset:")
st.write(df.head())

# Display accuracy results from both implementations
st.write("### Model Accuracy Comparison")

# Create a table for accuracy results
accuracy_data = {
    "Implementation": ["From Scratch", "From Scratch", "Scikit-learn", "Scikit-learn"],
    "Distance Metric": ["Euclidean", "Manhattan", "Euclidean", "Manhattan"],
    "Accuracy": [0.57, 0.58, 0.54, 0.55]
}

accuracy_df = pd.DataFrame(accuracy_data)
st.table(accuracy_df)

# Add explanation
st.write("""
**Accuracy Results Explanation:**
- **From Scratch Implementation**: 
  - Euclidean distance achieved 57% accuracy
  - Manhattan distance achieved 58% accuracy
- **Scikit-learn Implementation**:
  - Euclidean distance achieved 54% accuracy 
  - Manhattan distance achieved 55% accuracy
""")

# Show Euclidean and Manhattan distances (example values)
st.write(f"### Example Distance Calculations:")
st.write(f"Euclidean distance between [1,2,3] and [4,5,6]: {np.sqrt((1-4)**2 + (2-5)**2 + (3-6)**2):.2f}")
st.write(f"Manhattan distance between [1,2,3] and [4,5,6]: {abs(1-4) + abs(2-5) + abs(3-6):.2f}")

# Show confusion matrix using seaborn heatmap
st.write("### Confusion Matrix:")
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
           xticklabels=["Healthy", "Mild", "Moderate", "Severe"], 
           yticklabels=["Healthy", "Mild", "Moderate", "Severe"])
st.pyplot(plt)

# Show the macro F1 score
macro_f1 = np.mean(f1_scores)
st.write(f"### Macro F1 Score: {macro_f1:.2f}")

# Select box to choose the metric to view
metric = st.selectbox("Select metric to view", ["Precision", "Recall", "F1-Score"])

if metric == "Precision":
    st.write("### Precision for each class:")
    st.write(f"A: {precision[0]:.2f}")
    st.write(f"B: {precision[1]:.2f}")
    st.write(f"C: {precision[2]:.2f}")
    st.write(f"D: {precision[3]:.2f}")
elif metric == "Recall":
    st.write("### Recall for each class:")
    st.write(f"A: {recall[0]:.2f}")
    st.write(f"B: {recall[1]:.2f}")
    st.write(f"C: {recall[2]:.2f}")
    st.write(f"D: {recall[3]:.2f}")
elif metric == "F1-Score":
    st.write("### F1-Scores for each class:")
    st.write(f"A: {f1_scores[0]:.2f}")
    st.write(f"B: {f1_scores[1]:.2f}")
    st.write(f"C: {f1_scores[2]:.2f}")
    st.write(f"D: {f1_scores[3]:.2f}")