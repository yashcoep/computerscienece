import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier

# Load the feature-engineered dataset
data = pd.read_csv('feature_engineered_data.csv')

# Separate features and target
X = data.drop('Sentiment Label', axis=1)  # Features
y = data['Sentiment Label']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
logistic_regression = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Fit models
logistic_regression.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
mlp.fit(X_train, y_train)

# Initialize voting classifier
voting_classifier = VotingClassifier(estimators=[
    ('lr', logistic_regression),
    ('rf', random_forest),
    ('mlp', mlp)
], voting='hard')

# Fit voting classifier
voting_classifier.fit(X_train, y_train)

# Evaluate models
models = {
    'Logistic Regression': logistic_regression,
    'Random Forest': random_forest,
    'Multi-Layer Perceptron': mlp,
    'Voting Classifier': voting_classifier
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'Model: {name}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-Score: {f1:.2f}')
    print()
