import pandas as pd
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(comment):
    """
    Preprocesses the text by removing stop words, punctuation, and converting to lowercase.
    """
    doc = nlp(comment)
    processed_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    processed_comment = ' '.join(processed_tokens)
    return processed_comment

def preprocess_comments(input_file, output_file):
    """
    Preprocesses the comments from the input file and saves the processed data to the output file.
    """
    # Load the comments dataset
    comments_df = pd.read_csv(input_file)

    # Apply preprocessing to the 'Comment' column
    comments_df['Processed_Comment'] = comments_df['Comment'].apply(preprocess_text)

    # Save processed data to a new CSV file
    comments_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # Input and output file paths
    input_file = 'putin_tucker.csv'
    output_file = 'processed_comments.csv'

    # Preprocess comments and save to CSV
    preprocess_comments(input_file, output_file)
