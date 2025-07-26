"""
Model Evaluation Script for Personalized Recommendation System
Implements RMSE, MAE, Precision@K, and other evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Comprehensive evaluation of recommendation models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_rating_prediction(self, model, test_df, model_name):
        """Evaluate rating prediction accuracy"""
        print(f"Evaluating {model_name}...")
        
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                predicted_rating = model.predict(user_id, movie_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except:
                # Skip if prediction fails
                continue
        
        if len(predictions) == 0:
            print(f"No valid predictions for {model_name}")
            return {}
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
        
        results = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'n_predictions': len(predictions)
        }
        
        self.results[model_name] = results
        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        
        return results
    
    def evaluate_ranking_metrics(self, model, test_df, train_df, movies_df, model_name, k_values=[5, 10, 20]):
        """Evaluate ranking metrics like Precision@K, Recall@K, NDCG@K"""
        print(f"Evaluating ranking metrics for {model_name}...")
        
        ranking_results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            # Group by user
            for user_id in test_df['userId'].unique():
                user_test = test_df[test_df['userId'] == user_id]
                user_train = train_df[train_df['userId'] == user_id]
                
                if len(user_test) == 0:
                    continue
                
                # Get recommendations
                try:
                    if hasattr(model, 'recommend'):
                        recommendations = model.recommend(user_id, k * 2)  # Get more recommendations
                    elif hasattr(model, 'recommend_for_user'):
                        recommendations = model.recommend_for_user(user_id, train_df, k * 2)
                    else:
                        continue
                    
                    recommended_items = list(recommendations.keys())[:k]
                    
                    # Get relevant items (highly rated in test set)
                    relevant_items = set(user_test[user_test['rating'] >= 4]['movieId'].values)
                    
                    # Calculate metrics
                    precision = self._calculate_precision_at_k(recommended_items, relevant_items, k)
                    recall = self._calculate_recall_at_k(recommended_items, relevant_items, len(relevant_items))
                    ndcg = self._calculate_ndcg_at_k(recommended_items, user_test, k)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    ndcg_scores.append(ndcg)
                    
                except Exception as e:
                    continue
            
            if len(precision_scores) > 0:
                ranking_results[f'Precision@{k}'] = np.mean(precision_scores)
                ranking_results[f'Recall@{k}'] = np.mean(recall_scores)
                ranking_results[f'NDCG@{k}'] = np.mean(ndcg_scores)
        
        # Add to overall results
        self.results[model_name].update(ranking_results)
        
        print(f"{model_name} ranking metrics:")
        for metric, value in ranking_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return ranking_results
    
    def _calculate_precision_at_k(self, recommended_items, relevant_items, k):
        """Calculate Precision@K"""
        if k == 0:
            return 0
        
        relevant_recommended = len(set(recommended_items) & relevant_items)
        return relevant_recommended / k
    
    def _calculate_recall_at_k(self, recommended_items, relevant_items, total_relevant):
        """Calculate Recall@K"""
        if total_relevant == 0:
            return 0
        
        relevant_recommended = len(set(recommended_items) & relevant_items)
        return relevant_recommended / total_relevant
    
    def _calculate_ndcg_at_k(self, recommended_items, user_test, k):
        """Calculate NDCG@K"""
        if k == 0:
            return 0
        
        # Create relevance scores
        relevance_scores = {}
        for _, row in user_test.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            relevance_scores[movie_id] = rating
        
        # Calculate DCG
        dcg = 0
        for i, item in enumerate(recommended_items[:k]):
            relevance = relevance_scores.get(item, 0)
            dcg += relevance / np.log2(i + 2)  # log2(i+2) because i starts at 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)
        idcg = 0
        for i in range(min(k, len(ideal_relevance))):
            idcg += ideal_relevance[i] / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_diversity(self, model, train_df, movies_df, model_name, n_users=100, k=10):
        """Evaluate recommendation diversity"""
        print(f"Evaluating diversity for {model_name}...")
        
        user_sample = train_df['userId'].unique()[:n_users]
        all_recommendations = []
        
        for user_id in user_sample:
            try:
                if hasattr(model, 'recommend'):
                    recommendations = model.recommend(user_id, k)
                elif hasattr(model, 'recommend_for_user'):
                    recommendations = model.recommend_for_user(user_id, train_df, k)
                else:
                    continue
                
                recommended_items = list(recommendations.keys())
                all_recommendations.extend(recommended_items)
                
            except:
                continue
        
        if len(all_recommendations) == 0:
            return {}
        
        # Calculate diversity metrics
        total_items = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        
        # Calculate genre diversity
        genre_diversity = self._calculate_genre_diversity(all_recommendations, movies_df)
        
        diversity_results = {
            'Total_Unique_Items': total_items,
            'Total_Recommendations': total_recommendations,
            'Coverage_Ratio': total_items / len(movies_df),
            'Genre_Diversity': genre_diversity
        }
        
        self.results[model_name].update(diversity_results)
        
        print(f"{model_name} diversity metrics:")
        for metric, value in diversity_results.items():
            print(f"  {metric}: {value:.4f}")
        
        return diversity_results
    
    def _calculate_genre_diversity(self, recommended_items, movies_df):
        """Calculate genre diversity of recommendations"""
        genre_counts = defaultdict(int)
        total_movies = 0
        
        for movie_id in recommended_items:
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if len(movie_row) > 0:
                genres = movie_row.iloc[0]['genres']
                if pd.notna(genres):
                    for genre in genres.split('|'):
                        genre_counts[genre] += 1
                    total_movies += 1
        
        if total_movies == 0:
            return 0
        
        # Calculate Shannon diversity index
        proportions = [count / total_movies for count in genre_counts.values()]
        diversity = -sum(p * np.log(p) for p in proportions if p > 0)
        
        return diversity
    
    def compare_models(self, models_dict, test_df, train_df, movies_df):
        """Compare multiple models"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        comparison_results = {}
        
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            
            # Evaluate rating prediction
            rating_results = self.evaluate_rating_prediction(model, test_df, model_name)
            
            # Evaluate ranking metrics
            ranking_results = self.evaluate_ranking_metrics(model, test_df, train_df, movies_df, model_name)
            
            # Evaluate diversity
            diversity_results = self.evaluate_diversity(model, train_df, movies_df, model_name)
            
            comparison_results[model_name] = self.results[model_name]
        
        return comparison_results
    
    def create_comparison_table(self):
        """Create a comparison table of all models"""
        if not self.results:
            print("No results to compare")
            return
        
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame(self.results).T
        
        # Round numeric values
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns
        comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(comparison_df)
        
        return comparison_df
    
    def plot_results(self, save_path="results/model_comparison.png"):
        """Plot comparison results"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # RMSE and MAE
        metrics = ['RMSE', 'MAE']
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[0, i], color='skyblue')
                axes[0, i].set_title(f'{metric} Comparison')
                axes[0, i].set_ylabel(metric)
                axes[0, i].tick_params(axis='x', rotation=45)
        
        # Precision@10 and Recall@10
        ranking_metrics = ['Precision@10', 'Recall@10']
        for i, metric in enumerate(ranking_metrics):
            if metric in comparison_df.columns:
                comparison_df[metric].plot(kind='bar', ax=axes[1, i], color='lightcoral')
                axes[1, i].set_title(f'{metric} Comparison')
                axes[1, i].set_ylabel(metric)
                axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Results plot saved to {save_path}")
    
    def generate_report(self, output_path="results/evaluation_report.txt"):
        """Generate comprehensive evaluation report"""
        if not self.results:
            print("No results to report")
            return
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("RECOMMENDATION SYSTEM EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name}:\n")
                for metric, value in results.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n")
            
            # Find best model for each metric
            comparison_df = pd.DataFrame(self.results).T
            
            best_models = {}
            for metric in comparison_df.columns:
                if metric in ['RMSE', 'MAE']:
                    best_models[metric] = comparison_df[metric].idxmin()
                else:
                    best_models[metric] = comparison_df[metric].idxmax()
            
            f.write("\nBest performing models:\n")
            for metric, model in best_models.items():
                f.write(f"  {metric}: {model}\n")
        
        print(f"Evaluation report saved to {output_path}")

def main():
    """Test the evaluation system"""
    # Load sample data
    from src.data_processing.preprocess import DataPreprocessor
    from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
    from src.models.matrix_factorization import SVDRecommender, NMFRecommender
    from src.models.content_based import ContentBasedRecommender
    
    # Load data
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    
    if data is None:
        print("No data available. Please run preprocessing first.")
        return
    
    train_df = data['train']
    test_df = data['test']
    movies_df = data['movies']
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator()
    
    # Train models
    models = {
        'User-based CF': UserBasedCF(n_neighbors=20),
        'Item-based CF': ItemBasedCF(n_neighbors=20),
        'SVD': SVDRecommender(n_factors=50),
        'NMF': NMFRecommender(n_factors=50),
        'Content-based': ContentBasedRecommender()
    }
    
    # Fit models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == 'Content-based':
            model.fit(train_df, movies_df)
        else:
            model.fit(train_df)
    
    # Evaluate models
    comparison_results = evaluator.compare_models(models, test_df, train_df, movies_df)
    
    # Generate report
    evaluator.create_comparison_table()
    evaluator.plot_results()
    evaluator.generate_report()
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main() 