#!/usr/bin/env python3
"""
Quick Start Script for Personalized Recommendation System
Run this script to get started with the complete recommendation system
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner"""
    print("="*80)
    print("🎬 PERSONALIZED RECOMMENDATION SYSTEM")
    print("="*80)
    print("A comprehensive machine learning-based recommendation engine")
    print("with multiple algorithms and interactive web dashboard")
    print("="*80)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def run_pipeline():
    """Run the complete recommendation pipeline"""
    print("\n🚀 Running complete recommendation pipeline...")
    
    try:
        # Import and run the main pipeline
        from src.main import main as run_main
        run_main()
        return True
    except Exception as e:
        print(f"❌ Error running pipeline: {str(e)}")
        return False

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("\n🌐 Starting web dashboard...")
    print("The dashboard will open in your browser automatically.")
    print("If it doesn't open, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/dashboard/app.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {str(e)}")

def show_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "="*50)
        print("📋 MAIN MENU")
        print("="*50)
        print("1. Run Complete Pipeline (Data → Models → Evaluation)")
        print("2. Start Web Dashboard")
        print("3. Run Individual Components")
        print("4. Show Project Information")
        print("5. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            if run_pipeline():
                print("\n✅ Pipeline completed successfully!")
                print("Check the 'results' folder for outputs.")
            else:
                print("\n❌ Pipeline failed. Please check the error messages.")
        
        elif choice == "2":
            run_dashboard()
        
        elif choice == "3":
            show_individual_components()
        
        elif choice == "4":
            show_project_info()
        
        elif choice == "5":
            print("\n👋 Thank you for using the Recommendation System!")
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")

def show_individual_components():
    """Show menu for individual components"""
    while True:
        print("\n" + "="*50)
        print("🔧 INDIVIDUAL COMPONENTS")
        print("="*50)
        print("1. Download Dataset")
        print("2. Preprocess Data")
        print("3. Train Models")
        print("4. Evaluate Models")
        print("5. Back to Main Menu")
        print("="*50)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n📥 Downloading dataset...")
            from src.data_processing.download_data import main as download_main
            download_main()
        
        elif choice == "2":
            print("\n🔧 Preprocessing data...")
            from src.data_processing.preprocess import main as preprocess_main
            preprocess_main()
        
        elif choice == "3":
            print("\n🤖 Training models...")
            # This would require a separate training script
            print("Training models requires preprocessed data.")
            print("Please run preprocessing first.")
        
        elif choice == "4":
            print("\n📊 Evaluating models...")
            from src.evaluation.evaluate_models import main as evaluate_main
            evaluate_main()
        
        elif choice == "5":
            break
        
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")

def show_project_info():
    """Show project information"""
    print("\n" + "="*60)
    print("📚 PROJECT INFORMATION")
    print("="*60)
    
    print("🎯 Project: Personalized Recommendation System")
    print("📅 Version: 1.0.0")
    print("🎬 Dataset: MovieLens")
    
    print("\n🔧 Implemented Algorithms:")
    print("  • User-based Collaborative Filtering")
    print("  • Item-based Collaborative Filtering")
    print("  • SVD Matrix Factorization")
    print("  • NMF Matrix Factorization")
    print("  • Content-based Filtering (TF-IDF)")
    print("  • Genre-based Filtering")
    print("  • Hybrid Approaches")
    
    print("\n📊 Evaluation Metrics:")
    print("  • RMSE (Root Mean Square Error)")
    print("  • MAE (Mean Absolute Error)")
    print("  • Precision@K")
    print("  • Recall@K")
    print("  • NDCG@K")
    print("  • Diversity Metrics")
    
    print("\n🌐 Features:")
    print("  • Interactive Web Dashboard")
    print("  • Real-time Recommendations")
    print("  • Comprehensive Evaluation")
    print("  • Visualization and Reporting")
    print("  • Modular Architecture")
    
    print("\n📁 Project Structure:")
    print("  src/")
    print("  ├── data_processing/     # Data collection & preprocessing")
    print("  ├── models/             # Recommendation algorithms")
    print("  ├── evaluation/         # Model evaluation metrics")
    print("  ├── dashboard/          # Web interface")
    print("  └── main.py            # Complete pipeline")
    print("  notebooks/              # Jupyter notebooks")
    print("  results/                # Output files")
    print("  data/                   # Dataset storage")
    
    print("\n🚀 Quick Start:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run pipeline: python quick_start.py")
    print("  3. Start dashboard: streamlit run src/dashboard/app.py")
    
    print("\n📖 Documentation:")
    print("  • README.md - Project overview and setup")
    print("  • notebooks/ - Detailed analysis and examples")
    print("  • src/ - Source code documentation")
    
    print("="*60)

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Show menu
    show_menu()

if __name__ == "__main__":
    main() 