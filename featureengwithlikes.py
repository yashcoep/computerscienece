import pandas as pd
from textblob import TextBlob

# Load the dataset
data = pd.read_csv('processed_comments.csv')

# Calculate sentiment scores using TextBlob
sentiment_scores = data['Processed_Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Integrate likes and reply count as weighted factors
# You can adjust the weights based on your preference or domain knowledge
weighted_likes = data['Likes'] * 0.7
weighted_reply_count = data['Reply Count'] * 0.3

# Create additional features
comment_length = data['Processed_Comment'].apply(len)
publication_timestamps = pd.to_datetime(data['Published At'])
hour_of_day = publication_timestamps.dt.hour

# Create a new DataFrame with the calculated features
features_df = pd.DataFrame({
    'Sentiment Score': sentiment_scores,
    'Weighted Likes': weighted_likes,
    'Weighted Reply Count': weighted_reply_count,
    'Comment Length': comment_length,
    'Hour of Day': hour_of_day
})

# Generate sentiment labels based on sentiment scores
# You can define your own threshold for positive and negative sentiments
sentiment_labels = ['Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral' for score in sentiment_scores]

# Add sentiment labels to the DataFrame
features_df['Sentiment Label'] = sentiment_labels

# Save the feature-engineered DataFrame to a new CSV file
features_df.to_csv('feature_engineered_data.csv', index=False)

# Display the first few rows of the feature-engineered DataFrame
print(features_df.head())
