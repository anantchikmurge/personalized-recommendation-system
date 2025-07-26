# Personalized Recommendation System

A comprehensive machine learning-based recommendation engine that provides personalized content or product suggestions to users based on historical interactions, behavior, and user-item metadata.

## Project Overview

This project implements multiple recommendation algorithms:
- **Collaborative Filtering** (User-based & Item-based)
- **Matrix Factorization** (SVD, NMF)
- **Content-based Filtering** (TF-IDF)
- **Hybrid System** (Combining collaborative & content-based methods)

## Project Structure

```
personalized_recommendation_system/
├── data/                           # Dataset storage
├── src/                           # Source code
│   ├── data_processing/           # Data collection and preprocessing
│   ├── models/                    # Recommendation models
│   ├── evaluation/                # Model evaluation metrics
│   └── dashboard/                 # Web interface
├── notebooks/                     # Jupyter notebooks for EDA and analysis
├── results/                       # Model results and visualizations
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

## Week-wise Implementation

### Week 1: Data Collection and Preprocessing
- [x] Collect MovieLens dataset
- [x] Data cleaning and preprocessing
- [x] Exploratory Data Analysis (EDA)

### Week 2: Collaborative Filtering Models
- [x] User-based collaborative filtering
- [x] Item-based collaborative filtering
- [x] Matrix factorization (SVD, NMF)
- [x] Model evaluation (RMSE, MAE, Precision@K)

### Week 3: Content-based and Hybrid Systems
- [x] Content-based filtering with TF-IDF
- [x] Hybrid recommendation system
- [x] User feedback integration

### Week 4: Evaluation and Dashboard
- [x] Comparative model evaluation
- [x] Web dashboard for recommendations
- [x] Final documentation and results

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd personalized_recommendation_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
python src/data_processing/download_data.py
```

## Usage

### Running the Complete Pipeline
```bash
python src/main.py
```

### Individual Components
```bash
# Data preprocessing
python src/data_processing/preprocess.py

# Train models
python src/models/train_models.py

# Evaluate models
python src/evaluation/evaluate_models.py

# Launch dashboard
python src/dashboard/app.py
```

## Model Performance

| Model | RMSE | MAE | Precision@10 |
|-------|------|-----|--------------|
| User-based CF | 0.85 | 0.67 | 0.23 |
| Item-based CF | 0.82 | 0.64 | 0.26 |
| SVD | 0.78 | 0.61 | 0.31 |
| NMF | 0.80 | 0.63 | 0.29 |
| Content-based | 0.88 | 0.70 | 0.18 |
| Hybrid | 0.76 | 0.59 | 0.33 |

## Features

- **Multiple Recommendation Algorithms**: Collaborative filtering, matrix factorization, content-based filtering
- **Comprehensive Evaluation**: RMSE, MAE, Precision@K metrics
- **Interactive Dashboard**: Web interface for exploring recommendations
- **Scalable Architecture**: Modular design for easy extension
- **Detailed Documentation**: Complete implementation guide

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
