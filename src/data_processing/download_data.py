"""
Data Download Script for Personalized Recommendation System
Downloads MovieLens dataset and prepares it for processing
"""

import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class DataDownloader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.base_url = "https://files.grouplens.org/datasets/movielens"
        self.dataset_size = "ml-latest-small"  # Using small dataset for faster processing
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def download_dataset(self):
        """Download MovieLens dataset"""
        print("Downloading MovieLens dataset...")
        
        try:
            # Download ratings
            ratings_url = f"{self.base_url}/{self.dataset_size}.zip"
            zip_path = os.path.join(self.data_dir, f"{self.dataset_size}.zip")
            
            # Download with progress bar
            response = requests.get(ratings_url, stream=True, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as file, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            # Extract the zip file
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Remove the zip file
            os.remove(zip_path)
            
            print("Dataset downloaded successfully!")
            return self.load_data()
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_data(self):
        """Load the downloaded data"""
        try:
            dataset_path = os.path.join(self.data_dir, self.dataset_size)
            
            # Check if dataset exists
            if not os.path.exists(dataset_path):
                print(f"Dataset path not found: {dataset_path}")
                return None
            
            # Load ratings
            ratings_path = os.path.join(dataset_path, 'ratings.csv')
            if not os.path.exists(ratings_path):
                print(f"Ratings file not found: {ratings_path}")
                return None
                
            ratings_df = pd.read_csv(ratings_path)
            
            # Load movies
            movies_path = os.path.join(dataset_path, 'movies.csv')
            if not os.path.exists(movies_path):
                print(f"Movies file not found: {movies_path}")
                return None
                
            movies_df = pd.read_csv(movies_path)
            
            # Load tags (if available)
            tags_path = os.path.join(dataset_path, 'tags.csv')
            if os.path.exists(tags_path):
                tags_df = pd.read_csv(tags_path)
            else:
                tags_df = pd.DataFrame()
            
            # Load links (if available)
            links_path = os.path.join(dataset_path, 'links.csv')
            if os.path.exists(links_path):
                links_df = pd.read_csv(links_path)
            else:
                links_df = pd.DataFrame()
            
            return {
                'ratings': ratings_df,
                'movies': movies_df,
                'tags': tags_df,
                'links': links_df
            }
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_sample_data(self):
        """Create a smaller sample dataset for testing"""
        print("Creating sample dataset for testing...")
        
        try:
            # Generate synthetic data
            np.random.seed(42)
            n_users = 1000
            n_movies = 500
            n_ratings = 10000
            
            # Generate user IDs
            user_ids = np.random.randint(1, n_users + 1, n_ratings)
            
            # Generate movie IDs
            movie_ids = np.random.randint(1, n_movies + 1, n_ratings)
            
            # Generate ratings (1-5 scale)
            ratings = np.random.randint(1, 6, n_ratings)
            
            # Generate timestamps
            timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
            
            # Create ratings DataFrame
            ratings_df = pd.DataFrame({
                'userId': user_ids,
                'movieId': movie_ids,
                'rating': ratings,
                'timestamp': timestamps
            })
            
            # Create movies DataFrame
            movie_titles = [f"Movie {i}" for i in range(1, n_movies + 1)]
            genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
            movie_genres = np.random.choice(genres, n_movies)
            
            movies_df = pd.DataFrame({
                'movieId': range(1, n_movies + 1),
                'title': movie_titles,
                'genres': movie_genres
            })
            
            # Save to CSV files
            ratings_df.to_csv(os.path.join(self.data_dir, 'sample_ratings.csv'), index=False)
            movies_df.to_csv(os.path.join(self.data_dir, 'sample_movies.csv'), index=False)
            
            print("Sample dataset created successfully!")
            return {
                'ratings': ratings_df,
                'movies': movies_df,
                'tags': pd.DataFrame(),
                'links': pd.DataFrame()
            }
            
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return None

def main():
    """Main function to download and prepare data"""
    downloader = DataDownloader()
    
    try:
        # Try to download the real dataset
        data = downloader.download_dataset()
        if data is not None:
            print("Real MovieLens dataset loaded successfully!")
        else:
            print("Failed to download real dataset, creating sample data...")
            data = downloader.create_sample_data()
            
    except Exception as e:
        print(f"Error in main: {e}")
        print("Creating sample dataset instead...")
        data = downloader.create_sample_data()
    
    if data is None:
        print("Failed to create any dataset!")
        return None
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of ratings: {len(data['ratings'])}")
    print(f"Number of users: {data['ratings']['userId'].nunique()}")
    print(f"Number of movies: {data['ratings']['movieId'].nunique()}")
    print(f"Average rating: {data['ratings']['rating'].mean():.2f}")
    
    return data

if __name__ == "__main__":
    main() 