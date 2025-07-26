"""
Content-based Filtering Models for Personalized Recommendation System
Implements TF-IDF based content filtering using movie metadata
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    """Content-based recommendation using TF-IDF and metadata"""
    
    def __init__(self, max_features=1000, n_gram_range=(1, 2)):
        self.max_features = max_features
        self.n_gram_range = n_gram_range
        self.tfidf_vectorizer = None
        self.movie_features = None
        self.movie_similarity_matrix = None
        self.movies_df = None
        self.fitted = False
    
    def fit(self, ratings_df, movies_df):
        """Fit the content-based model"""
        print("Training Content-based Recommender...")
        
        self.movies_df = movies_df.copy()
        
        # Prepare movie content features
        movie_content = self._prepare_movie_content(movies_df)
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.n_gram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        self.movie_features = self.tfidf_vectorizer.fit_transform(movie_content)
        
        # Calculate movie similarity matrix
        self.movie_similarity_matrix = cosine_similarity(self.movie_features)
        
        # Convert to DataFrame for easier indexing
        self.movie_similarity_matrix = pd.DataFrame(
            self.movie_similarity_matrix,
            index=movies_df['movieId'],
            columns=movies_df['movieId']
        )
        
        self.fitted = True
        print("Content-based model training completed!")
        
        return self
    
    def _prepare_movie_content(self, movies_df):
        """Prepare movie content for TF-IDF vectorization"""
        content_list = []
        
        for _, movie in movies_df.iterrows():
            # Combine title and genres
            title = str(movie.get('title', ''))
            genres = str(movie.get('genres', ''))
            
            # Clean and combine content
            content = f"{title} {genres}".lower()
            content = re.sub(r'[^\w\s]', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            content_list.append(content)
        
        return content_list
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get similar movies based on content"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if movie_id not in self.movie_similarity_matrix.index:
            return {}
        
        # Get similarity scores for the movie
        similarities = self.movie_similarity_matrix[movie_id].sort_values(ascending=False)[1:n_similar+1]
        
        return similarities.to_dict()
    
    def recommend_for_user(self, user_id, ratings_df, n_recommendations=10):
        """Get content-based recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get user's rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            # Return most popular movies for new users
            return self._get_popular_movies(n_recommendations)
        
        # Calculate user profile based on rated movies
        user_profile = self._calculate_user_profile(user_ratings)
        
        # Get recommendations based on user profile
        recommendations = self._get_recommendations_from_profile(user_profile, user_ratings, n_recommendations)
        
        return recommendations
    
    def _calculate_user_profile(self, user_ratings):
        """Calculate user profile based on rated movies"""
        profile = np.zeros(self.movie_features.shape[1])
        
        for _, rating in user_ratings.iterrows():
            movie_id = rating['movieId']
            rating_score = rating['rating']
            
            if movie_id in self.movies_df['movieId'].values:
                movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
                movie_vector = self.movie_features[movie_idx].toarray().flatten()
                
                # Weight by rating (higher ratings get more weight)
                profile += movie_vector * rating_score
        
        # Normalize profile
        if np.sum(profile) > 0:
            profile = profile / np.sum(profile)
        
        return profile
    
    def _get_recommendations_from_profile(self, user_profile, user_ratings, n_recommendations):
        """Get recommendations based on user profile"""
        # Get user's rated movie IDs
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        # Calculate similarity between user profile and all movies
        similarities = []
        
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movieId']
            
            # Skip movies the user has already rated
            if movie_id in rated_movie_ids:
                continue
            
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            movie_vector = self.movie_features[movie_idx].toarray().flatten()
            
            # Calculate cosine similarity
            similarity = np.dot(user_profile, movie_vector) / (np.linalg.norm(user_profile) * np.linalg.norm(movie_vector) + 1e-8)
            similarities.append((movie_id, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return dict(similarities[:n_recommendations])
    
    def _get_popular_movies(self, n_recommendations):
        """Get most popular movies for new users"""
        # Return movies with highest average ratings
        movie_means = self.movies_df.set_index('movieId')['title'].to_dict()
        return dict(list(movie_means.items())[:n_recommendations])

class GenreBasedRecommender:
    """Genre-based content filtering"""
    
    def __init__(self):
        self.movies_df = None
        self.genre_matrix = None
        self.genre_similarity_matrix = None
        self.fitted = False
    
    def fit(self, ratings_df, movies_df):
        """Fit the genre-based recommender"""
        print("Training Genre-based Recommender...")
        
        self.movies_df = movies_df.copy()
        
        # Process genres
        self._process_genres()
        
        # Create genre similarity matrix
        genre_columns = [col for col in self.movies_df.columns if col.startswith('genre_')]
        if genre_columns:
            self.genre_matrix = self.movies_df[genre_columns].values
            self.movie_ids = self.movies_df['movieId'].values
        else:
            # Fallback: create simple genre similarity
            self.genre_matrix = None
            self.movie_ids = self.movies_df['movieId'].values
        
        self.fitted = True
        print("Genre-based model training completed!")
    
    def _process_genres(self):
        """Process genres into binary columns."""
        all_genres = set()
        for genres_str in self.movies_df['genres'].dropna():
            if pd.notna(genres_str):
                genre_list = [g.strip() for g in genres_str.split('|')]
                all_genres.update(genre_list)
        
        # Create binary columns for each unique genre
        for genre in all_genres:
            self.movies_df[f'genre_{genre}'] = self.movies_df['genres'].apply(
                lambda x: 1 if pd.notna(x) and genre in x.split('|') else 0
            )
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """Get similar movies based on genres"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if movie_id not in self.movie_ids:
            return {}
        
        movie_idx = np.where(self.movie_ids == movie_id)[0][0]
        
        if self.genre_matrix is None:
            # Fallback to simple cosine similarity if genre matrix is not available
            genre_scores = {}
            for i, other_movie_id in enumerate(self.movie_ids):
                if other_movie_id == movie_id:
                    continue
                other_movie_idx = np.where(self.movie_ids == other_movie_id)[0][0]
                similarity = cosine_similarity(self.genre_matrix[movie_idx].reshape(1, -1), self.genre_matrix[other_movie_idx].reshape(1, -1))[0][0]
                genre_scores[other_movie_id] = similarity
        else:
            # Use genre matrix for similarity
            genre_scores = {}
            for i, other_movie_id in enumerate(self.movie_ids):
                if other_movie_id == movie_id:
                    continue
                other_movie_idx = np.where(self.movie_ids == other_movie_id)[0][0]
                similarity = cosine_similarity(self.genre_matrix[movie_idx].reshape(1, -1), self.genre_matrix[other_movie_idx].reshape(1, -1))[0][0]
                genre_scores[other_movie_id] = similarity
        
        # Sort by similarity and return top N
        sorted_genre_scores = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_genre_scores[:n_similar])
    
    def recommend_for_user(self, user_id, ratings_df, n_recommendations=10):
        """Get genre-based recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get user's rated movies
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return self._get_popular_movies(n_recommendations)
        
        # Calculate user's genre preferences
        user_genre_prefs = self._calculate_user_genre_preferences(user_ratings)
        
        # Get recommendations
        recommendations = self._get_recommendations_from_genre_prefs(user_genre_prefs, user_ratings, n_recommendations)
        
        return recommendations
    
    def _calculate_user_genre_preferences(self, user_ratings):
        """Calculate user's genre preferences"""
        genre_scores = {}
        
        for _, rating in user_ratings.iterrows():
            movie_id = rating['movieId']
            rating_score = rating['rating']
            
            movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_row) > 0:
                genres = movie_row.iloc[0]['genres']
                if pd.notna(genres):
                    for genre in genres.split('|'):
                        if genre not in genre_scores:
                            genre_scores[genre] = 0
                        genre_scores[genre] += rating_score
        
        return genre_scores
    
    def _get_recommendations_from_genre_prefs(self, user_genre_prefs, user_ratings, n_recommendations):
        """Get recommendations based on genre preferences"""
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        movie_scores = []
        
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movieId']
            
            if movie_id in rated_movie_ids:
                continue
            
            genres = movie['genres']
            if pd.notna(genres):
                score = 0
                for genre in genres.split('|'):
                    if genre in user_genre_prefs:
                        score += user_genre_prefs[genre]
                
                movie_scores.append((movie_id, score))
        
        # Sort by score and return top N
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        return dict(movie_scores[:n_recommendations])
    
    def _get_popular_movies(self, n_recommendations):
        """Get most popular movies for new users"""
        # Return movies with highest average ratings
        movie_means = self.movies_df.set_index('movieId')['title'].to_dict()
        return dict(list(movie_means.items())[:n_recommendations])

class HybridContentRecommender:
    """Hybrid content-based recommender combining multiple approaches"""
    
    def __init__(self, content_weight=0.6, genre_weight=0.4):
        self.content_weight = content_weight
        self.genre_weight = genre_weight
        self.content_model = ContentBasedRecommender()
        self.genre_model = GenreBasedRecommender()
        self.fitted = False
    
    def fit(self, ratings_df, movies_df):
        """Fit the hybrid model"""
        print("Training Hybrid Content-based Recommender...")
        
        # Fit both models
        self.content_model.fit(ratings_df, movies_df)
        self.genre_model.fit(ratings_df, movies_df) # Changed to ratings_df, movies_df
        
        self.fitted = True
        print("Hybrid content-based model training completed!")
        
        return self
    
    def recommend_for_user(self, user_id, ratings_df, n_recommendations=10):
        """Get hybrid recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        content_recs = self.content_model.recommend_for_user(user_id, ratings_df, n_recommendations * 2)
        genre_recs = self.genre_model.recommend_for_user(user_id, ratings_df, n_recommendations * 2)
        
        # Combine recommendations
        combined_scores = {}
        
        for movie_id, score in content_recs.items():
            combined_scores[movie_id] = self.content_weight * score
        
        for movie_id, score in genre_recs.items():
            if movie_id in combined_scores:
                combined_scores[movie_id] += self.genre_weight * score
            else:
                combined_scores[movie_id] = self.genre_weight * score
        
        # Sort and return top N
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_recs[:n_recommendations])

def main():
    """Test the content-based filtering models"""
    # Load sample data
    from src.data_processing.preprocess import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    
    if data is None:
        print("No data available. Please run preprocessing first.")
        return
    
    train_df = data['train']
    movies_df = data['movies']
    
    # Test Content-based CF
    print("\n" + "="*50)
    print("TESTING CONTENT-BASED FILTERING")
    print("="*50)
    
    content_model = ContentBasedRecommender()
    content_model.fit(train_df, movies_df)
    
    # Test recommendations
    test_user = train_df['userId'].iloc[0]
    recommendations = content_model.recommend_for_user(test_user, train_df, 5)
    print(f"Content-based recommendations for user {test_user}:")
    for movie_id, score in list(recommendations.items())[:5]:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f"  {movie_title}: {score:.3f}")
    
    # Test Genre-based CF
    print("\n" + "="*50)
    print("TESTING GENRE-BASED FILTERING")
    print("="*50)
    
    genre_model = GenreBasedRecommender()
    genre_model.fit(train_df, movies_df) # Changed to train_df, movies_df
    
    recommendations = genre_model.recommend_for_user(test_user, train_df, 5)
    print(f"Genre-based recommendations for user {test_user}:")
    for movie_id, score in list(recommendations.items())[:5]:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f"  {movie_title}: {score:.3f}")
    
    print("\nContent-based filtering models tested successfully!")

if __name__ == "__main__":
    main() 