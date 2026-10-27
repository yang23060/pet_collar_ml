import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the dataset
data = pd.read_csv("dog_activity.csv")

print("Dataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# Separate features (X) and labels (y)
X = data.drop("label", axis=1)
y = data["label"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nLabels:")
print(y.value_counts())


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)


# Train the model
model.fit(X_train, y_train)

print("\nModel training complete!")


# Save the trained model
joblib.dump(model, "dog_activity_model.pkl")

print("Model saved as dog_activity_model.pkl")


# Make predictions on the test data
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", f"{accuracy * 100:.2f}%")


# Confusion matrix
print("\nConfusion Matrix:")

labels = ["still", "moving", "rotating", "shaking"]

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print(matrix)


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=labels
    )
)