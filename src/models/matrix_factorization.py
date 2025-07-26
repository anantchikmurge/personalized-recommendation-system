"""
Matrix Factorization Models for Recommendation System
Implements SVD, NMF, and custom gradient descent-based matrix factorization
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SVDRecommender:
    """Singular Value Decomposition (SVD) based recommender"""
    
    def __init__(self, n_factors=100, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.global_mean = 0
        self.fitted = False
    
    def fit(self, ratings_df):
        """Fit the SVD model"""
        print("Training SVD model...")
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        # Store mappings
        self.user_mapping = {user: idx for idx, user in enumerate(user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(user_item_matrix.columns)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Calculate global mean
        self.global_mean = ratings_df['rating'].mean()
        
        # Center the data
        centered_matrix = user_item_matrix - self.global_mean
        
        # Apply SVD
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.user_factors = self.svd.fit_transform(centered_matrix)
        self.item_factors = self.svd.components_.T
        
        self.fitted = True
        print(f"SVD model trained with {self.n_factors} factors")
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-item pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        prediction += self.global_mean
        
        return max(1, min(5, prediction))  # Clip to rating range
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            return {}
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(user_vector, self.item_factors.T)
        scores += self.global_mean
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = {}
        for idx in top_indices:
            movie_id = self.reverse_item_mapping[idx]
            recommendations[movie_id] = scores[idx]
        
        return recommendations

class NMFRecommender:
    """Non-negative Matrix Factorization (NMF) based recommender"""
    
    def __init__(self, n_factors=100, random_state=42, max_iter=200):
        self.n_factors = n_factors
        self.random_state = random_state
        self.max_iter = max_iter
        self.nmf = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.fitted = False
    
    def fit(self, ratings_df):
        """Fit the NMF model"""
        print("Training NMF model...")
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        # Store mappings
        self.user_mapping = {user: idx for idx, user in enumerate(user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(user_item_matrix.columns)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Apply NMF
        self.nmf = NMF(n_components=self.n_factors, random_state=self.random_state, max_iter=self.max_iter)
        self.user_factors = self.nmf.fit_transform(user_item_matrix)
        self.item_factors = self.nmf.components_.T
        
        self.fitted = True
        print(f"NMF model trained with {self.n_factors} factors")
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-item pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return 0
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]
        
        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return max(1, min(5, prediction))  # Clip to rating range
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            return {}
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(user_vector, self.item_factors.T)
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = {}
        for idx in top_indices:
            movie_id = self.reverse_item_mapping[idx]
            recommendations[movie_id] = scores[idx]
        
        return recommendations

class MatrixFactorizationRecommender:
    """Custom gradient descent-based matrix factorization"""
    
    def __init__(self, n_factors=100, learning_rate=0.01, n_epochs=100,
                 regularization=0.1, random_state=42):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.fitted = False
    
    def fit(self, ratings_df):
        """Fit the custom matrix factorization model"""
        print("Training custom matrix factorization model...")
        
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating', 
            fill_value=0
        )
        
        # Store mappings
        self.user_mapping = {user: idx for idx, user in enumerate(user_item_matrix.index)}
        self.item_mapping = {item: idx for idx, item in enumerate(user_item_matrix.columns)}
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        # Initialize parameters
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # Convert to sparse format for training
        train_data = []
        for _, row in ratings_df.iterrows():
            user_idx = self.user_mapping.get(row['userId'])
            item_idx = self.item_mapping.get(row['movieId'])
            if user_idx is not None and item_idx is not None:
                train_data.append((user_idx, item_idx, row['rating']))
        
        # Gradient descent
        for epoch in range(self.n_epochs):
            total_error = 0
            for user_idx, item_idx, rating in train_data:
                # Predict rating
                pred = (self.global_mean + 
                       self.user_biases[user_idx] + 
                       self.item_biases[item_idx] + 
                       np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
                
                # Calculate error
                error = rating - pred
                total_error += error ** 2
                
                # Update parameters
                self.user_biases[user_idx] += self.learning_rate * (error - self.regularization * self.user_biases[user_idx])
                self.item_biases[item_idx] += self.learning_rate * (error - self.regularization * self.item_biases[item_idx])
                
                # Update factors
                user_factor_grad = error * self.item_factors[item_idx] - self.regularization * self.user_factors[user_idx]
                item_factor_grad = error * self.user_factors[user_idx] - self.regularization * self.item_factors[item_idx]
                
                self.user_factors[user_idx] += self.learning_rate * user_factor_grad
                self.item_factors[item_idx] += self.learning_rate * item_factor_grad
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, RMSE: {np.sqrt(total_error / len(train_data)):.4f}")
        
        self.fitted = True
        print(f"Custom matrix factorization model trained with {self.n_factors} factors")
    
    def predict(self, user_id, movie_id):
        """Predict rating for a user-item pair"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return self.global_mean
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[movie_id]
        
        prediction = (self.global_mean + 
                     self.user_biases[user_idx] + 
                     self.item_biases[item_idx] + 
                     np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
        
        return max(1, min(5, prediction))  # Clip to rating range
    
    def recommend(self, user_id, n_recommendations=10):
        """Generate recommendations for a user"""
        if not self.fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_mapping:
            return {}
        
        user_idx = self.user_mapping[user_id]
        
        # Calculate scores for all items
        scores = (self.global_mean + 
                 self.user_biases[user_idx] + 
                 self.item_biases + 
                 np.dot(self.user_factors[user_idx], self.item_factors.T))
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = {}
        for idx in top_indices:
            movie_id = self.reverse_item_mapping[idx]
            recommendations[movie_id] = scores[idx]
        
        return recommendations 