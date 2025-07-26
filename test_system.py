#!/usr/bin/env python3
"""
Test Script for Personalized Recommendation System
Quick test to verify all components work correctly
"""

import os
import sys
import traceback

def test_imports():
    """Test if all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        # Test data processing imports
        from src.data_processing.download_data import DataDownloader
        from src.data_processing.preprocess import DataPreprocessor
        print("✅ Data processing imports successful")
        
        # Test model imports
        from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
        from src.models.matrix_factorization import SVDRecommender, NMFRecommender
        from src.models.content_based import ContentBasedRecommender, GenreBasedRecommender
        print("✅ Model imports successful")
        
        # Test evaluation imports
        from src.evaluation.evaluate_models import RecommendationEvaluator
        print("✅ Evaluation imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_download():
    """Test data download functionality"""
    print("\n📥 Testing data download...")
    
    try:
        from src.data_processing.download_data import DataDownloader
        
        downloader = DataDownloader()
        data = downloader.create_sample_data()
        
        if data is not None and 'ratings' in data and 'movies' in data:
            print(f"✅ Sample data created successfully!")
            print(f"   - Ratings: {len(data['ratings'])} records")
            print(f"   - Movies: {len(data['movies'])} records")
            return True
        else:
            print("❌ Failed to create sample data")
            return False
            
    except Exception as e:
        print(f"❌ Error in data download test: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing"""
    print("\n🔧 Testing data preprocessing...")
    
    try:
        from src.data_processing.preprocess import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline()
        
        if data is not None and 'train' in data and 'test' in data:
            print(f"✅ Data preprocessing successful!")
            print(f"   - Train set: {len(data['train'])} records")
            print(f"   - Test set: {len(data['test'])} records")
            print(f"   - Movies: {len(data['movies'])} records")
            return True
        else:
            print("❌ Failed to preprocess data")
            return False
            
    except Exception as e:
        print(f"❌ Error in preprocessing test: {e}")
        return False

def test_model_training():
    """Test model training"""
    print("\n🤖 Testing model training...")
    
    try:
        from src.data_processing.preprocess import DataPreprocessor
        from src.models.collaborative_filtering import UserBasedCF
        
        # Load data
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline()
        
        if data is None:
            print("❌ No data available for model training test")
            return False
        
        # Test a simple model
        model = UserBasedCF(n_neighbors=10)
        model.fit(data['train'])
        
        if hasattr(model, 'fitted') and model.fitted:
            print("✅ Model training successful!")
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in model training test: {e}")
        return False

def test_recommendations():
    """Test recommendation generation"""
    print("\n🎯 Testing recommendation generation...")
    
    try:
        from src.data_processing.preprocess import DataPreprocessor
        from src.models.collaborative_filtering import UserBasedCF
        
        # Load data
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline()
        
        if data is None:
            print("❌ No data available for recommendation test")
            return False
        
        # Train model
        model = UserBasedCF(n_neighbors=10)
        model.fit(data['train'])
        
        # Test recommendations
        user_id = data['train']['userId'].iloc[0]
        recommendations = model.recommend(user_id, 5)
        
        if recommendations and len(recommendations) > 0:
            print("✅ Recommendation generation successful!")
            print(f"   - Generated {len(recommendations)} recommendations")
            return True
        else:
            print("❌ Failed to generate recommendations")
            return False
            
    except Exception as e:
        print(f"❌ Error in recommendation test: {e}")
        return False

def test_evaluation():
    """Test evaluation functionality"""
    print("\n📊 Testing evaluation...")
    
    try:
        from src.data_processing.preprocess import DataPreprocessor
        from src.models.collaborative_filtering import UserBasedCF
        from src.evaluation.evaluate_models import RecommendationEvaluator
        
        # Load data
        preprocessor = DataPreprocessor()
        data = preprocessor.preprocess_pipeline()
        
        if data is None:
            print("❌ No data available for evaluation test")
            return False
        
        # Train model
        model = UserBasedCF(n_neighbors=10)
        model.fit(data['train'])
        
        # Test evaluation
        evaluator = RecommendationEvaluator()
        results = evaluator.evaluate_rating_prediction(model, data['test'], "Test Model")
        
        if results and 'RMSE' in results:
            print("✅ Evaluation successful!")
            print(f"   - RMSE: {results['RMSE']:.4f}")
            print(f"   - MAE: {results['MAE']:.4f}")
            return True
        else:
            print("❌ Evaluation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in evaluation test: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🧪 PERSONALIZED RECOMMENDATION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Download Test", test_data_download),
        ("Preprocessing Test", test_preprocessing),
        ("Model Training Test", test_model_training),
        ("Recommendation Test", test_recommendations),
        ("Evaluation Test", test_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("📋 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The system is working correctly.")
        print("You can now run the complete pipeline or start the dashboard.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("Make sure all dependencies are installed and the project structure is correct.")
    
    print("\n🚀 Next steps:")
    print("1. Run complete pipeline: python quick_start.py")
    print("2. Start dashboard: streamlit run src/dashboard/app.py")
    print("3. Explore notebooks: jupyter notebook notebooks/")

if __name__ == "__main__":
    main() 