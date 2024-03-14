import pandas as pd
import matplotlib.pyplot as plt

# Load the feature-engineered data
feature_engineered_data = pd.read_csv('feature_engineered_data_constrastive_sentiment.csv')

# Visualize sentiment distributions
plt.figure(figsize=(8, 6))
feature_engineered_data['Sentiment Label'].value_counts().plot(kind='bar', color=['green', 'yellow' ,'red', 'blue', 'purple'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Visualize sentiment trends over time
plt.figure(figsize=(10, 6))
feature_engineered_data.plot(x='Hour of Day', y='Sentiment Score', kind='scatter', color='blue')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Hour of Day')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualize relationship between sentiment and likes
plt.figure(figsize=(10, 6))
plt.scatter(feature_engineered_data['Weighted Likes'], feature_engineered_data['Sentiment Score'], alpha=0.5)
plt.title('Relationship Between Sentiment and Likes')
plt.xlabel('Weighted Likes')
plt.ylabel('Sentiment Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# Generate reports summarizing sentiment analysis findings and insights
# You can customize the report content based on your specific analysis and insights

# Example report content:
print("Summary Report:")
print("1. Sentiment Distribution:")
print(feature_engineered_data['Sentiment Label'].value_counts())
print("\n2. Sentiment Trends Over Time:")
print(feature_engineered_data.groupby('Hour of Day')['Sentiment Score'].mean())
print("\n3. Relationship Between Sentiment and Likes:")
print(feature_engineered_data[['Weighted Likes', 'Sentiment Score']].corr())

# Additional analysis and insights can be added to the report as needed
