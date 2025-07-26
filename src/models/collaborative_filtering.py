"""
Collaborative Filtering Models for Personalized Recommendation System
Implements user-based and item-based collaborative filtering
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class UserBasedCF:
    """User-based Collaborative Filtering"""
    
    def __init__(self, n_neighbors=50, similarity_metric='cosine'):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_means = None
        self.fitted = False
    
    def fit(self, ratings_df):
        """Fit the user-based CF model"""
        print("Training User-based Collaborative Filtering...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Calculate user means for normalization
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Normalize ratings by subtracting user means
        normalized_matrix = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Calculate user similarity matrix
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(normalized_matrix)
        elif self.similarity_metric == 'euclidean':
            self.user_similarity_matrix = 1 / (1 + euclidean_distances(normalized_matrix))
        else:
            raise ValueError("Similarity metric must be 'cosine' or 'euclidean'")
        
        # Convert to DataFrame for easier indexing
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.fitted = True
        print("User-based CF training completed!")
        
        return self
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_item_matrix.index:
            return self.user_means.mean()  # Return global mean for new users
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_means.mean()  # Return global mean for new movies
        
        # Get user's mean rating
        user_mean = self.user_means[user_id]
        
        # Find similar users who rated this movie
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:self.n_neighbors+1]
        
        # Get ratings for this movie from similar users
        movie_ratings = self.user_item_matrix[movie_id]
        similar_user_ratings = movie_ratings[similar_users.index]
        
        # Filter out users who haven't rated this movie
        rated_users = similar_user_ratings[similar_user_ratings != 0]
        
        if len(rated_users) == 0:
            return user_mean
        
        # Calculate weighted average
        weights = similar_users[rated_users.index]
        weighted_sum = (weights * (rated_users - self.user_means[rated_users.index])).sum()
        weight_sum = weights.sum()
        
        if weight_sum == 0:
            return user_mean
        
        prediction = user_mean + (weighted_sum / weight_sum)
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id, n_recommendations=10):
        """Get top N recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            # Return most popular movies for new users
            movie_means = self.user_item_matrix.mean()
            return movie_means.sort_values(ascending=False).head(n_recommendations)
        
        # Get user's unrated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = {}
        for movie_id in unrated_movies:
            predictions[movie_id] = self.predict(user_id, movie_id)
        
        # Sort by predicted rating and return top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_predictions[:n_recommendations])

class ItemBasedCF:
    """Item-based Collaborative Filtering"""
    
    def __init__(self, n_neighbors=50, similarity_metric='cosine'):
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.item_means = None
        self.fitted = False
    
    def fit(self, ratings_df):
        """Fit the item-based CF model"""
        print("Training Item-based Collaborative Filtering...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Calculate item means for normalization
        self.item_means = self.user_item_matrix.mean(axis=0)
        
        # Normalize ratings by subtracting item means
        normalized_matrix = self.user_item_matrix.sub(self.item_means, axis=1)
        
        # Calculate item similarity matrix
        if self.similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(normalized_matrix.T)
        elif self.similarity_metric == 'euclidean':
            self.item_similarity_matrix = 1 / (1 + euclidean_distances(normalized_matrix.T))
        else:
            raise ValueError("Similarity metric must be 'cosine' or 'euclidean'")
        
        # Convert to DataFrame for easier indexing
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        self.fitted = True
        print("Item-based CF training completed!")
        
        return self
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_item_matrix.index:
            return self.item_means.mean()  # Return global mean for new users
        
        if movie_id not in self.user_item_matrix.columns:
            return self.item_means.mean()  # Return global mean for new movies
        
        # Get item's mean rating
        item_mean = self.item_means[movie_id]
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings != 0]
        
        if len(rated_items) == 0:
            return item_mean
        
        # Find similar items to the target movie
        similar_items = self.item_similarity_matrix[movie_id].sort_values(ascending=False)[1:self.n_neighbors+1]
        
        # Get ratings for items the user has rated
        common_items = rated_items.index.intersection(similar_items.index)
        
        if len(common_items) == 0:
            return item_mean
        
        # Calculate weighted average
        weights = similar_items[common_items]
        weighted_sum = (weights * (rated_items[common_items] - self.item_means[common_items])).sum()
        weight_sum = weights.sum()
        
        if weight_sum == 0:
            return item_mean
        
        prediction = item_mean + (weighted_sum / weight_sum)
        
        # Clip to valid rating range
        return np.clip(prediction, 1, 5)
    
    def recommend(self, user_id, n_recommendations=10):
        """Get top N recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_item_matrix.index:
            # Return most popular movies for new users
            return self.item_means.sort_values(ascending=False).head(n_recommendations)
        
        # Get user's unrated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = {}
        for movie_id in unrated_movies:
            predictions[movie_id] = self.predict(user_id, movie_id)
        
        # Sort by predicted rating and return top N
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_predictions[:n_recommendations])

class KNNRecommender:
    """K-Nearest Neighbors based recommender"""
    
    def __init__(self, k=50, similarity_metric='cosine'):
        self.k = k
        self.similarity_metric = similarity_metric
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.fitted = False
    
    def fit(self, ratings_df, method='user'):
        """Fit the KNN model"""
        print(f"Training KNN-based {method} Collaborative Filtering...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        # Calculate similarity matrix
        if method == 'user':
            if self.similarity_metric == 'cosine':
                self.similarity_matrix = cosine_similarity(self.user_item_matrix)
            else:
                self.similarity_matrix = 1 / (1 + euclidean_distances(self.user_item_matrix))
            
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.index
            )
        else:  # item-based
            if self.similarity_metric == 'cosine':
                self.similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            else:
                self.similarity_matrix = 1 / (1 + euclidean_distances(self.user_item_matrix.T))
            
            self.similarity_matrix = pd.DataFrame(
                self.similarity_matrix,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        
        self.method = method
        self.fitted = True
        print(f"KNN {method}-based CF training completed!")
        
        return self
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.method == 'user':
            return self._predict_user_based(user_id, movie_id)
        else:
            return self._predict_item_based(user_id, movie_id)
    
    def _predict_user_based(self, user_id, movie_id):
        """User-based prediction"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.mean().mean()
        
        # Get similar users
        similar_users = self.similarity_matrix[user_id].sort_values(ascending=False)[1:self.k+1]
        
        # Get ratings for this movie from similar users
        movie_ratings = self.user_item_matrix[movie_id]
        similar_user_ratings = movie_ratings[similar_users.index]
        
        # Filter out users who haven't rated this movie
        rated_users = similar_user_ratings[similar_user_ratings != 0]
        
        if len(rated_users) == 0:
            return self.user_item_matrix[movie_id].mean()
        
        # Calculate weighted average
        weights = similar_users[rated_users.index]
        weighted_sum = (weights * rated_users).sum()
        weight_sum = weights.sum()
        
        return weighted_sum / weight_sum if weight_sum > 0 else self.user_item_matrix[movie_id].mean()
    
    def _predict_item_based(self, user_id, movie_id):
        """Item-based prediction"""
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.mean().mean()
        
        # Get user's rated items
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings != 0]
        
        if len(rated_items) == 0:
            return self.user_item_matrix[movie_id].mean()
        
        # Find similar items
        similar_items = self.similarity_matrix[movie_id].sort_values(ascending=False)[1:self.k+1]
        
        # Get common items
        common_items = rated_items.index.intersection(similar_items.index)
        
        if len(common_items) == 0:
            return self.user_item_matrix[movie_id].mean()
        
        # Calculate weighted average
        weights = similar_items[common_items]
        weighted_sum = (weights * rated_items[common_items]).sum()
        weight_sum = weights.sum()
        
        return weighted_sum / weight_sum if weight_sum > 0 else self.user_item_matrix[movie_id].mean()

def main():
    """Test the collaborative filtering models"""
    # Load sample data
    from src.data_processing.preprocess import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    
    if data is None:
        print("No data available. Please run preprocessing first.")
        return
    
    train_df = data['train']
    
    # Test User-based CF
    print("\n" + "="*50)
    print("TESTING USER-BASED COLLABORATIVE FILTERING")
    print("="*50)
    
    user_cf = UserBasedCF(n_neighbors=20)
    user_cf.fit(train_df)
    
    # Test predictions
    test_user = train_df['userId'].iloc[0]
    test_movie = train_df['movieId'].iloc[0]
    prediction = user_cf.predict(test_user, test_movie)
    print(f"Predicted rating for user {test_user}, movie {test_movie}: {prediction:.2f}")
    
    # Test Item-based CF
    print("\n" + "="*50)
    print("TESTING ITEM-BASED COLLABORATIVE FILTERING")
    print("="*50)
    
    item_cf = ItemBasedCF(n_neighbors=20)
    item_cf.fit(train_df)
    
    prediction = item_cf.predict(test_user, test_movie)
    print(f"Predicted rating for user {test_user}, movie {test_movie}: {prediction:.2f}")
    
    print("\nCollaborative filtering models tested successfully!")

if __name__ == "__main__":
    main() 