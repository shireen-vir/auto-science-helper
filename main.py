import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    Main function for auto-science-helper data science tool.

    This tool is designed to assist with basic data science tasks such as data loading, 
    preprocessing, and model training. It supports simple classification tasks using 
    a random forest classifier.
    """
    # Load dataset
    try:
        data = pd.read_csv('data.csv')
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    # Preprocess data
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()