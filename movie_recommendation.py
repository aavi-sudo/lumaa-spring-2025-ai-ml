import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

def clean_text(text):
    """
    Cleans the text by:
    - Converting it to lowercase
    - Removing special characters (e.g., punctuation)
    - Removing extra spaces

    This helps standardize text before processing.
    """
    if pd.isna(text):  # Handle cases where text is NaN
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def load_movie_data():
    """
    Loads the movie dataset from a CSV file.
    - Handles missing values by removing rows with NaN descriptions.
    - Cleans movie descriptions using the `clean_text` function.
    - Removes duplicate movie descriptions to avoid redundant data.

    Returns:
        - A cleaned pandas DataFrame containing 'title' and 'description'.
    """
    try:
        df = pd.read_csv('MOVIESS.csv')  # Load CSV file
        df = df[['title', 'description']].dropna().copy()  # Keep only 'title' and 'description', drop NaN values
        df['description'] = df['description'].apply(clean_text)  # Apply text cleaning
        df = df[df['description'].str.strip() != '']  # Remove empty descriptions
        df = df.drop_duplicates(subset=['title', 'description']).reset_index(drop=True)  # Remove duplicate descriptions
        return df
    except FileNotFoundError:
        print("Error: MOVIESS.csv not found! Make sure the file is in the correct directory.")
        exit()  # Exit if the dataset is missing

def remove_outliers(df):
    """
    Removes outliers based on text length.
    - Calculates mean and standard deviation of description lengths.
    - Removes descriptions that are too short or too long (beyond 2 standard deviations).

    This helps maintain high-quality recommendations.

    Returns:
        - A filtered DataFrame with outliers removed.
    """
    df = df.copy()  # Prevent modifying the original DataFrame
    df['text_length'] = df['description'].apply(len)  # Compute length of each description

    mean_len = df['text_length'].mean()  # Calculate mean length
    std_len = df['text_length'].std()  # Calculate standard deviation

    # Keep descriptions within a reasonable length range (within 2 standard deviations)
    df = df[(df['text_length'] > mean_len - 2 * std_len) & (df['text_length'] < mean_len + 2 * std_len)]
    
    return df.drop(columns=['text_length'])  # Drop the extra column after filtering

def find_similar_movies(user_input, df, vectorizer):
    """
    Finds the top 5 movies similar to the user's input.
    - Converts movie descriptions and user input into TF-IDF vectors.
    - Computes cosine similarity to find the closest matches.

    Returns:
        - A DataFrame containing the top 5 similar movie recommendations.
    """
    if df.empty:  # Check if dataset is empty
        print("No data available to compare.")
        return pd.DataFrame()

    # Convert user input into a TF-IDF vector
    user_vector = vectorizer.transform([clean_text(user_input)])

    # Compute similarity scores between user input and all movie descriptions
    scores = cosine_similarity(user_vector, vectorizer.transform(df['description']))[0]

    if np.all(scores == 0):  # If no meaningful similarity found
        print("\nNo exact matches found, but here are some random recommendations:\n")
        return df.sample(n=5)  # Pick 5 random movies without a fixed seed

    # Get indices of the top 5 most similar movies
    top_matches = scores.argsort()[-5:][::-1]

    # Return results with similarity scores
    df_result = df.iloc[top_matches].copy()
    df_result['similarity_score'] = scores[top_matches]  # Add similarity score column
    return df_result

def main():
    """
    Runs the recommender system with all essential steps.
    - Loads the movie dataset.
    - Removes outliers to improve data quality.
    - Trains the TF-IDF vectorizer (only once).
    - Continuously accepts user input to suggest movies.
    - Allows users to exit by typing "exit".
    """
    print("Welcome to the Simple Movie Recommender!\n")

    df = load_movie_data()  # Load and clean dataset
    df = remove_outliers(df)  # Remove outlier descriptions

    # Train TF-IDF Vectorizer **only once** on movie descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(df['description'])

    while True:
        # Get user input for movie preferences
        user_input = input("\nTell me what kind of movies you like (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':  # Exit condition
            print("Thank you for using the recommender! 😊")
            break

        # Get top 5 movie recommendations
        recommendations = find_similar_movies(user_input, df, vectorizer)

        if not recommendations.empty:
            print("\nTop recommended movies for you:")
            for i, row in recommendations.iterrows():
                if 'similarity_score' in row:  # Only show similarity if available
                    print(f"{i+1}. {row['title']} (Similarity Score: {row['similarity_score']:.2f}) - {row['description'][:100]}...")
                else:
                    print(f"{i+1}. {row['title']} - {row['description'][:100]}...")  # No similarity score for random suggestions
        else:
            print("\nNo suitable recommendations found.")

if __name__ == "__main__":
    main()
