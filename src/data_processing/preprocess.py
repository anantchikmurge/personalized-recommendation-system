"""
Data Preprocessing Script for Personalized Recommendation System
Cleans and prepares data for recommendation models
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.label_encoders = {}
        
    def load_data(self):
        """Load data from CSV files"""
        try:
            # Try to load real MovieLens data
            ratings_path = os.path.join(self.data_dir, "ml-latest-small", "ratings.csv")
            movies_path = os.path.join(self.data_dir, "ml-latest-small", "movies.csv")
            
            if os.path.exists(ratings_path) and os.path.exists(movies_path):
                ratings_df = pd.read_csv(ratings_path)
                movies_df = pd.read_csv(movies_path)
                print("Loaded real MovieLens dataset")
            else:
                # Load sample data
                ratings_df = pd.read_csv(os.path.join(self.data_dir, "sample_ratings.csv"))
                movies_df = pd.read_csv(os.path.join(self.data_dir, "sample_movies.csv"))
                print("Loaded sample dataset")
                
        except FileNotFoundError:
            print("No data files found. Please run download_data.py first.")
            return None, None
            
        return ratings_df, movies_df
    
    def clean_ratings(self, ratings_df):
        """Clean and preprocess ratings data"""
        print("Cleaning ratings data...")
        
        # Remove duplicates
        initial_count = len(ratings_df)
        ratings_df = ratings_df.drop_duplicates()
        print(f"Removed {initial_count - len(ratings_df)} duplicate ratings")
        
        # Remove invalid ratings (outside 1-5 range)
        ratings_df = ratings_df[ratings_df['rating'].between(1, 5)]
        print(f"Removed ratings outside valid range (1-5)")
        
        # Convert timestamp to datetime
        if 'timestamp' in ratings_df.columns:
            ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
            ratings_df['year'] = ratings_df['timestamp'].dt.year
            ratings_df['month'] = ratings_df['timestamp'].dt.month
        
        # Add rating categories
        ratings_df['rating_category'] = pd.cut(
            ratings_df['rating'], 
            bins=[0, 2, 3, 4, 5], 
            labels=['Low', 'Medium', 'Good', 'Excellent']
        )
        
        return ratings_df
    
    def clean_movies(self, movies_df):
        """Clean and preprocess movies data"""
        print("Cleaning movies data...")
        
        # Remove movies with missing titles
        movies_df = movies_df.dropna(subset=['title'])
        
        # Clean movie titles
        movies_df['title_clean'] = movies_df['title'].apply(self._clean_title)
        
        # Extract year from title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
        movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
        
        # Process genres
        if 'genres' in movies_df.columns:
            movies_df = self._process_genres(movies_df)
        
        return movies_df
    
    def _clean_title(self, title):
        """Clean movie title"""
        if pd.isna(title):
            return title
        
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)\s*', '', str(title))
        
        # Remove special characters but keep spaces
        title = re.sub(r'[^\w\s]', '', title)
        
        # Convert to lowercase
        title = title.lower().strip()
        
        return title
    
    def _process_genres(self, movies_df):
        """Process movie genres"""
        # Split genres and create genre columns
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            genre_list = genres.split('|')
            all_genres.update(genre_list)
        
        # Create binary columns for each genre
        for genre in all_genres:
            movies_df[f'genre_{genre.lower().replace(" ", "_")}'] = movies_df['genres'].str.contains(genre, na=False).astype(int)
        
        # Create genre count
        movies_df['genre_count'] = movies_df['genres'].str.count(r'\|') + 1
        movies_df['genre_count'] = movies_df['genre_count'].fillna(0)
        
        return movies_df
    
    def create_user_item_matrix(self, ratings_df):
        """Create user-item rating matrix"""
        print("Creating user-item matrix...")
        
        # Create pivot table
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Calculate sparsity
        sparsity = 1 - (user_item_matrix != 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        print(f"Matrix sparsity: {sparsity:.4f}")
        
        return user_item_matrix
    
    def encode_categorical_features(self, df, columns):
        """Encode categorical features"""
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
    
    def create_features(self, ratings_df, movies_df):
        """Create additional features for recommendation models"""
        print("Creating additional features...")
        
        # User features
        user_stats = ratings_df.groupby('userId')['rating'].agg(['count', 'mean', 'std']).reset_index()
        user_stats.columns = ['userId', 'user_rating_count', 'user_avg_rating', 'user_rating_std']
        user_stats = user_stats.fillna(0)
        
        # Movie features
        movie_stats = ratings_df.groupby('movieId')['rating'].agg(['count', 'mean', 'std']).reset_index()
        movie_stats.columns = ['movieId', 'movie_rating_count', 'movie_avg_rating', 'movie_rating_std']
        movie_stats = movie_stats.fillna(0)
        
        # Merge with original data
        ratings_df = ratings_df.merge(user_stats, on='userId', how='left')
        ratings_df = ratings_df.merge(movie_stats, on='movieId', how='left')
        
        # Add interaction features
        ratings_df['rating_deviation'] = ratings_df['rating'] - ratings_df['movie_avg_rating']
        ratings_df['user_rating_centered'] = ratings_df['rating'] - ratings_df.groupby('userId')['rating'].transform('mean')
        
        # Fill NaN values - handle categorical columns
        for col in ratings_df.columns:
            if ratings_df[col].dtype.name == 'category':
                ratings_df[col] = ratings_df[col].astype(str)
            ratings_df[col] = ratings_df[col].fillna(0)
        
        return ratings_df, user_stats, movie_stats
    
    def split_data(self, ratings_df, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print("Splitting data into train and test sets...")
        
        # Sort by timestamp if available
        if 'timestamp' in ratings_df.columns:
            ratings_df = ratings_df.sort_values('timestamp')
        
        # Create train/test split
        np.random.seed(random_state)
        test_indices = np.random.choice(
            len(ratings_df), 
            size=int(len(ratings_df) * test_size), 
            replace=False
        )
        
        train_mask = ~ratings_df.index.isin(test_indices)
        train_df = ratings_df[train_mask].copy()
        test_df = ratings_df[~train_mask].copy()
        
        print(f"Train set: {len(train_df)} ratings")
        print(f"Test set: {len(test_df)} ratings")
        
        return train_df, test_df
    
    def get_data_statistics(self, ratings_df, movies_df):
        """Get comprehensive data statistics"""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"\nRatings Dataset:")
        print(f"Total ratings: {len(ratings_df):,}")
        print(f"Unique users: {ratings_df['userId'].nunique():,}")
        print(f"Unique movies: {ratings_df['movieId'].nunique():,}")
        print(f"Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"Rating distribution:")
        print(ratings_df['rating'].value_counts().sort_index())
        
        print(f"\nMovies Dataset:")
        print(f"Total movies: {len(movies_df):,}")
        if 'genres' in movies_df.columns:
            print(f"Unique genres: {movies_df['genres'].nunique()}")
        
        print(f"\nSparsity Analysis:")
        total_possible_ratings = ratings_df['userId'].nunique() * ratings_df['movieId'].nunique()
        actual_ratings = len(ratings_df)
        sparsity = 1 - (actual_ratings / total_possible_ratings)
        print(f"Matrix sparsity: {sparsity:.4f}")
        
        print(f"\nUser Activity:")
        user_activity = ratings_df.groupby('userId')['rating'].count()
        print(f"Average ratings per user: {user_activity.mean():.1f}")
        print(f"Median ratings per user: {user_activity.median():.1f}")
        print(f"Most active user: {user_activity.max()} ratings")
        
        print(f"\nMovie Popularity:")
        movie_popularity = ratings_df.groupby('movieId')['rating'].count()
        print(f"Average ratings per movie: {movie_popularity.mean():.1f}")
        print(f"Median ratings per movie: {movie_popularity.median():.1f}")
        print(f"Most popular movie: {movie_popularity.max()} ratings")
        
        print("="*50)
    
    def save_processed_data(self, train_df, test_df, movies_df, output_dir="data/processed"):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, "train_ratings.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_ratings.csv"), index=False)
        movies_df.to_csv(os.path.join(output_dir, "movies_processed.csv"), index=False)
        
        print(f"Processed data saved to {output_dir}")
    
    def preprocess_pipeline(self):
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing pipeline...")
        
        # Load data
        ratings_df, movies_df = self.load_data()
        if ratings_df is None:
            return None
        
        # Clean data
        ratings_df = self.clean_ratings(ratings_df)
        movies_df = self.clean_movies(movies_df)
        
        # Create features
        ratings_df, user_features, movie_features = self.create_features(ratings_df, movies_df)
        
        # Get statistics
        self.get_data_statistics(ratings_df, movies_df)
        
        # Split data
        train_df, test_df = self.split_data(ratings_df)
        
        # Save processed data
        self.save_processed_data(train_df, test_df, movies_df)
        
        return {
            'train': train_df,
            'test': test_df,
            'movies': movies_df,
            'user_features': user_features,
            'movie_features': movie_features
        }

def main():
    """Main preprocessing function"""
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline()
    
    if processed_data:
        print("\nPreprocessing completed successfully!")
        return processed_data
    else:
        print("Preprocessing failed. Please check your data files.")
        return None

if __name__ == "__main__":
    main() 