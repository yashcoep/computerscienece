import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load the feature-engineered dataset
data = pd.read_csv('feature_engineered_data.csv')

# Separate features and target
X = data.drop('Sentiment Label', axis=1)  # Features
y = data['Sentiment Label']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the models
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Make predictions on the test data
logistic_regression_preds = logistic_regression.predict(X_test)
random_forest_preds = random_forest.predict(X_test)
mlp_preds = mlp.predict(X_test)

# Error analysis
logistic_regression_errors = X_test[y_test != logistic_regression_preds]
random_forest_errors = X_test[y_test != random_forest_preds]
mlp_errors = X_test[y_test != mlp_preds]

# Print error analysis results
print("Logistic Regression Misclassified Instances:")
print(logistic_regression_errors)

print("\nRandom Forest Misclassified Instances:")
print(random_forest_errors)

print("\nMulti-Layer Perceptron Misclassified Instances:")
print(mlp_errors)
