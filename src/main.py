"""
Main Pipeline for Personalized Recommendation System
Orchestrates the complete workflow from data loading to model evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.download_data import DataDownloader
from data_processing.preprocess import DataPreprocessor
from models.collaborative_filtering import UserBasedCF, ItemBasedCF
from models.matrix_factorization import SVDRecommender, NMFRecommender
from models.content_based import ContentBasedRecommender, GenreBasedRecommender
from evaluation.evaluate_models import RecommendationEvaluator

class RecommendationPipeline:
    """Complete recommendation system pipeline"""
    
    def __init__(self):
        self.data = None
        self.models = {}
        self.evaluator = RecommendationEvaluator()
        self.results = {}
    
    def run_complete_pipeline(self):
        """Run the complete recommendation system pipeline"""
        print("="*80)
        print("PERSONALIZED RECOMMENDATION SYSTEM PIPELINE")
        print("="*80)
        
        # Step 1: Data Collection
        print("\n📥 STEP 1: Data Collection")
        print("-" * 40)
        self._collect_data()
        
        # Step 2: Data Preprocessing
        print("\n🔧 STEP 2: Data Preprocessing")
        print("-" * 40)
        self._preprocess_data()
        
        # Step 3: Model Training
        print("\n🤖 STEP 3: Model Training")
        print("-" * 40)
        self._train_models()
        
        # Step 4: Model Evaluation
        print("\n📊 STEP 4: Model Evaluation")
        print("-" * 40)
        self._evaluate_models()
        
        # Step 5: Generate Results
        print("\n📈 STEP 5: Generate Results")
        print("-" * 40)
        self._generate_results()
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def _collect_data(self):
        """Collect and download data"""
        try:
            downloader = DataDownloader()
            data = downloader.download_dataset()
            
            if data is None:
                print("Creating sample dataset...")
                data = downloader.create_sample_data()
            
            print("✅ Data collection completed!")
            return data
            
        except Exception as e:
            print(f"❌ Error in data collection: {str(e)}")
            return None
    
    def _preprocess_data(self):
        """Preprocess the data"""
        try:
            preprocessor = DataPreprocessor()
            self.data = preprocessor.preprocess_pipeline()
            
            if self.data is None:
                print("❌ Data preprocessing failed!")
                return False
            
            print("✅ Data preprocessing completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in data preprocessing: {str(e)}")
            return False
    
    def _train_models(self):
        """Train all recommendation models"""
        try:
            train_df = self.data['train']
            movies_df = self.data['movies']
            
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
                print(f"Training {name}...")
                try:
                    if name in ['Content-based', 'Genre-based']:
                        model.fit(train_df, movies_df)
                    else:
                        model.fit(train_df)
                    print(f"✅ {name} trained successfully!")
                except Exception as e:
                    print(f"❌ Error training {name}: {str(e)}")
            
            print("✅ Model training completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
            return False
    
    def _evaluate_models(self):
        """Evaluate all models"""
        try:
            test_df = self.data['test']
            train_df = self.data['train']
            movies_df = self.data['movies']
            
            # Run evaluation
            self.results = self.evaluator.compare_models(
                self.models, test_df, train_df, movies_df
            )
            
            print("✅ Model evaluation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in model evaluation: {str(e)}")
            return False
    
    def _generate_results(self):
        """Generate final results and reports"""
        try:
            # Create results directory
            os.makedirs("results", exist_ok=True)
            
            # Generate comparison table
            comparison_df = self.evaluator.create_comparison_table()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save comparison table
            comparison_df.to_csv(f"results/model_comparison_{timestamp}.csv")
            
            # Generate plots
            self.evaluator.plot_results(f"results/model_comparison_{timestamp}.png")
            
            # Generate report
            self.evaluator.generate_report(f"results/evaluation_report_{timestamp}.txt")
            
            # Save sample recommendations
            self._save_sample_recommendations()
            
            print("✅ Results generation completed!")
            return True
            
        except Exception as e:
            print(f"❌ Error in results generation: {str(e)}")
            return False
    
    def _save_sample_recommendations(self):
        """Save sample recommendations for each model"""
        try:
            train_df = self.data['train']
            movies_df = self.data['movies']
            
            # Get a sample user
            sample_user = train_df['userId'].iloc[0]
            
            recommendations_data = {}
            
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'recommend'):
                        recommendations = model.recommend(sample_user, 10)
                    elif hasattr(model, 'recommend_for_user'):
                        recommendations = model.recommend_for_user(sample_user, train_df, 10)
                    else:
                        continue
                    
                    # Convert to readable format
                    rec_list = []
                    for movie_id, score in recommendations.items():
                        movie_info = movies_df[movies_df['movieId'] == movie_id]
                        if len(movie_info) > 0:
                            title = movie_info.iloc[0]['title']
                            genres = movie_info.iloc[0].get('genres', 'Unknown')
                            rec_list.append({
                                'movie_id': movie_id,
                                'title': title,
                                'genres': genres,
                                'score': score
                            })
                    
                    recommendations_data[model_name] = rec_list
                    
                except Exception as e:
                    print(f"Error getting recommendations for {model_name}: {str(e)}")
            
            # Save recommendations
            import json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(f"results/sample_recommendations_{timestamp}.json", 'w') as f:
                json.dump(recommendations_data, f, indent=2)
            
            print("✅ Sample recommendations saved!")
            
        except Exception as e:
            print(f"❌ Error saving sample recommendations: {str(e)}")
    
    def get_best_model(self, metric='RMSE'):
        """Get the best performing model for a given metric"""
        if not self.results:
            print("No evaluation results available")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        
        if metric not in comparison_df.columns:
            print(f"Metric {metric} not found in results")
            return None
        
        if metric in ['RMSE', 'MAE']:
            best_model = comparison_df[metric].idxmin()
        else:
            best_model = comparison_df[metric].idxmax()
        
        return best_model
    
    def get_recommendations(self, user_id, model_name=None, n_recommendations=10):
        """Get recommendations for a user"""
        if not self.models:
            print("No models available")
            return None
        
        if model_name is None:
            # Use best model based on RMSE
            model_name = self.get_best_model('RMSE')
            if model_name is None:
                model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        try:
            model = self.models[model_name]
            train_df = self.data['train']
            
            if hasattr(model, 'recommend'):
                recommendations = model.recommend(user_id, n_recommendations)
            elif hasattr(model, 'recommend_for_user'):
                recommendations = model.recommend_for_user(user_id, train_df, n_recommendations)
            else:
                print("Model doesn't support recommendations")
                return None
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return None

def main():
    """Main function to run the complete pipeline"""
    pipeline = RecommendationPipeline()
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()
    
    # Show summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    if pipeline.results:
        comparison_df = pd.DataFrame(pipeline.results).T
        print("\nModel Performance Summary:")
        print(comparison_df[['RMSE', 'MAE', 'Precision@10', 'Recall@10']].round(4))
        
        # Show best models
        print("\nBest Models:")
        for metric in ['RMSE', 'MAE', 'Precision@10', 'Recall@10']:
            if metric in comparison_df.columns:
                best_model = pipeline.get_best_model(metric)
                print(f"  {metric}: {best_model}")
    
    print("\nResults saved in the 'results' directory!")
    print("To run the web dashboard, use: streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main() 