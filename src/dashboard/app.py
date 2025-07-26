"""
Streamlit Dashboard for Personalized Recommendation System
Interactive web interface for exploring recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.data_processing.preprocess import DataPreprocessor
    from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
    from src.models.matrix_factorization import SVDRecommender, NMFRecommender
    from src.models.content_based import ContentBasedRecommender, GenreBasedRecommender
    from src.evaluation.evaluate_models import RecommendationEvaluator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure all dependencies are installed and the project structure is correct.")
    st.stop()

class RecommendationDashboard:
    """Streamlit dashboard for recommendation system"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.evaluator = RecommendationEvaluator()
        
    def load_data(self):
        """Load and prepare data"""
        with st.spinner("Loading data..."):
            try:
                preprocessor = DataPreprocessor()
                self.data = preprocessor.preprocess_pipeline()
                
                if self.data is None:
                    st.error("Failed to load data. Please check your data files.")
                    return False
                
                st.success("Data loaded successfully!")
                return True
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False
    
    def train_models(self):
        """Train all recommendation models"""
        with st.spinner("Training models..."):
            try:
                train_df = self.data['train']
                movies_df = self.data['movies']
                
                # Try to load pre-trained models first
                try:
                    import pickle
                    if os.path.exists('models/trained_models.pkl'):
                        with open('models/trained_models.pkl', 'rb') as f:
                            self.models = pickle.load(f)
                        st.success("✅ Pre-trained models loaded successfully!")
                        return
                except Exception as e:
                    st.info("No pre-trained models found, training new models...")
                
                # Initialize models
                self.models = {
                    'User-based CF': UserBasedCF(n_neighbors=20),
                    'Item-based CF': ItemBasedCF(n_neighbors=20),
                    'SVD': SVDRecommender(n_factors=50),
                    'NMF': NMFRecommender(n_factors=50),
                    'Content-based': ContentBasedRecommender(),
                    'Genre-based': GenreBasedRecommender()
                }
                
                # Train models
                for name, model in self.models.items():
                    with st.spinner(f"Training {name}..."):
                        try:
                            if name in ['Content-based', 'Genre-based']:
                                model.fit(train_df, movies_df)
                            else:
                                model.fit(train_df)
                            st.success(f"{name} trained successfully!")
                        except Exception as e:
                            st.error(f"Error training {name}: {str(e)}")
                            
            except Exception as e:
                st.error(f"Error in model training: {str(e)}")
    
    def main(self):
        """Main dashboard function"""
        st.set_page_config(
            page_title="Personalized Recommendation System",
            page_icon="🎬",
            layout="wide"
        )
        
        st.title("🎬 Personalized Movie Recommendation System")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Data Overview", "Model Training", "Recommendations", "Model Evaluation", "About"]
        )
        
        # Load data if not already loaded
        if self.data is None:
            if not self.load_data():
                st.error("Failed to load data. Please check your data files and try again.")
                return
        
        # Navigation
        if page == "Data Overview":
            self.show_data_overview()
        elif page == "Model Training":
            self.show_model_training()
        elif page == "Recommendations":
            self.show_recommendations()
        elif page == "Model Evaluation":
            self.show_model_evaluation()
        elif page == "About":
            self.show_about()
    
    def show_data_overview(self):
        """Show data overview and statistics"""
        st.header("📊 Data Overview")
        
        try:
            train_df = self.data['train']
            test_df = self.data['test']
            movies_df = self.data['movies']
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Ratings", f"{len(train_df):,}")
            with col2:
                st.metric("Unique Users", f"{train_df['userId'].nunique():,}")
            with col3:
                st.metric("Unique Movies", f"{train_df['movieId'].nunique():,}")
            with col4:
                st.metric("Average Rating", f"{train_df['rating'].mean():.2f}")
            
            # Rating distribution
            st.subheader("Rating Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            train_df['rating'].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Ratings')
            ax.set_xlabel('Rating')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
            # User activity
            st.subheader("User Activity")
            user_activity = train_df.groupby('userId')['rating'].count()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                user_activity.hist(bins=50, ax=ax)
                ax.set_title('Distribution of Ratings per User')
                ax.set_xlabel('Number of Ratings')
                ax.set_ylabel('Number of Users')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                movie_popularity = train_df.groupby('movieId')['rating'].count()
                movie_popularity.hist(bins=50, ax=ax)
                ax.set_title('Distribution of Ratings per Movie')
                ax.set_xlabel('Number of Ratings')
                ax.set_ylabel('Number of Movies')
                st.pyplot(fig)
            
            # Genre analysis
            if 'genres' in movies_df.columns:
                st.subheader("Genre Analysis")
                
                # Get genre counts
                all_genres = []
                for genres in movies_df['genres'].dropna():
                    all_genres.extend(genres.split('|'))
                
                genre_counts = pd.Series(all_genres).value_counts()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                genre_counts.head(10).plot(kind='bar', ax=ax)
                ax.set_title('Top 10 Movie Genres')
                ax.set_xlabel('Genre')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error in data overview: {str(e)}")
    
    def show_model_training(self):
        """Show model training interface"""
        st.header("🤖 Model Training")
        
        if st.button("Train All Models"):
            self.train_models()
            st.success("All models trained successfully!")
        
        # Show model status
        if self.models:
            st.subheader("Model Status")
            for name, model in self.models.items():
                status = "✅ Trained" if hasattr(model, 'fitted') and model.fitted else "❌ Not Trained"
                st.write(f"{name}: {status}")
    
    def show_recommendations(self):
        """Show recommendation interface"""
        st.header("🎯 Get Recommendations")
        
        if not self.models:
            st.warning("Please train models first in the Model Training page.")
            return
        
        try:
            # User selection
            train_df = self.data['train']
            movies_df = self.data['movies']
            
            user_ids = train_df['userId'].unique()
            selected_user = st.selectbox("Select a user:", user_ids)
            
            # Model selection
            model_names = list(self.models.keys())
            selected_model = st.selectbox("Select a model:", model_names)
            
            # Number of recommendations
            n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
            
            if st.button("Get Recommendations"):
                with st.spinner("Generating recommendations..."):
                    try:
                        model = self.models[selected_model]
                        
                        # Get recommendations
                        if hasattr(model, 'recommend'):
                            recommendations = model.recommend(selected_user, n_recommendations)
                        elif hasattr(model, 'recommend_for_user'):
                            recommendations = model.recommend_for_user(selected_user, train_df, n_recommendations)
                        else:
                            st.error("Model doesn't support recommendations")
                            return
                        
                        # Display recommendations
                        st.subheader(f"Top {n_recommendations} Recommendations for User {selected_user}")
                        
                        # Create recommendation table
                        rec_data = []
                        for movie_id, score in recommendations.items():
                            movie_info = movies_df[movies_df['movieId'] == movie_id]
                            if len(movie_info) > 0:
                                title = movie_info.iloc[0]['title']
                                genres = movie_info.iloc[0].get('genres', 'Unknown')
                                rec_data.append({
                                    'Movie': title,
                                    'Genres': genres,
                                    'Score': f"{score:.3f}"
                                })
                        
                        rec_df = pd.DataFrame(rec_data)
                        st.dataframe(rec_df, use_container_width=True)
                        
                        # Show user's rated movies
                        user_ratings = train_df[train_df['userId'] == selected_user]
                        if len(user_ratings) > 0:
                            st.subheader("User's Rated Movies")
                            
                            rated_data = []
                            for _, rating in user_ratings.iterrows():
                                movie_info = movies_df[movies_df['movieId'] == rating['movieId']]
                                if len(movie_info) > 0:
                                    title = movie_info.iloc[0]['title']
                                    genres = movie_info.iloc[0].get('genres', 'Unknown')
                                    rated_data.append({
                                        'Movie': title,
                                        'Genres': genres,
                                        'Rating': rating['rating']
                                    })
                            
                            rated_df = pd.DataFrame(rated_data)
                            st.dataframe(rated_df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error in recommendations: {str(e)}")
    
    def show_model_evaluation(self):
        """Show model evaluation results"""
        st.header("📈 Model Evaluation")
        
        if not self.models:
            st.warning("Please train models first in the Model Training page.")
            return
        
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating models..."):
                try:
                    # Run evaluation
                    test_df = self.data['test']
                    train_df = self.data['train']
                    movies_df = self.data['movies']
                    
                    comparison_results = self.evaluator.compare_models(
                        self.models, test_df, train_df, movies_df
                    )
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    
                    # Create comparison table
                    comparison_df = self.evaluator.create_comparison_table()
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Create plots
                    st.subheader("Performance Comparison")
                    
                    # RMSE and MAE comparison
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    metrics = ['RMSE', 'MAE']
                    for i, metric in enumerate(metrics):
                        if metric in comparison_df.columns:
                            comparison_df[metric].plot(kind='bar', ax=axes[0, i])
                            axes[0, i].set_title(f'{metric} Comparison')
                            axes[0, i].tick_params(axis='x', rotation=45)
                    
                    # Precision@10 and Recall@10 comparison
                    ranking_metrics = ['Precision@10', 'Recall@10']
                    for i, metric in enumerate(ranking_metrics):
                        if metric in comparison_df.columns:
                            comparison_df[metric].plot(kind='bar', ax=axes[1, i])
                            axes[1, i].set_title(f'{metric} Comparison')
                            axes[1, i].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Generate report
                    self.evaluator.generate_report()
                    st.success("Evaluation completed! Check the results folder for detailed report.")
                
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
    
    def show_about(self):
        """Show about page"""
        st.header("ℹ️ About")
        
        st.markdown("""
        ## Personalized Recommendation System
        
        This is a comprehensive recommendation system that implements multiple algorithms:
        
        ### Models Implemented:
        - **Collaborative Filtering**: User-based and Item-based approaches
        - **Matrix Factorization**: SVD and NMF techniques
        - **Content-based Filtering**: TF-IDF and genre-based approaches
        
        ### Features:
        - Interactive web dashboard
        - Comprehensive model evaluation
        - Real-time recommendations
        - Performance metrics (RMSE, MAE, Precision@K, Recall@K)
        
        ### Dataset:
        - MovieLens dataset
        - Movie ratings and metadata
        - User-item interaction data
        
        ### Technologies:
        - Python 3.8+
        - Streamlit for web interface
        - Scikit-learn for machine learning
        - Pandas and NumPy for data processing
        - Matplotlib and Plotly for visualization
        """)
        
        st.markdown("---")
        st.markdown("**Project Structure:**")
        st.code("""
        personalized_recommendation_system/
        ├── data/                           # Dataset storage
        ├── src/                           # Source code
        │   ├── data_processing/           # Data collection and preprocessing
        │   ├── models/                    # Recommendation models
        │   ├── evaluation/                # Model evaluation metrics
        │   └── dashboard/                 # Web interface
        ├── notebooks/                     # Jupyter notebooks
        ├── results/                       # Model results and visualizations
        ├── requirements.txt               # Python dependencies
        └── README.md                     # Project documentation
        """)

def main():
    """Main function to run the dashboard"""
    try:
        dashboard = RecommendationDashboard()
        dashboard.main()
    except Exception as e:
        st.error(f"Error starting dashboard: {str(e)}")
        st.error("Please check your installation and dependencies.")

if __name__ == "__main__":
    main() 