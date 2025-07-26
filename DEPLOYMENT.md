# 🚀 Deployment Guide for Streamlit Cloud

## 📋 Prerequisites

1. **GitHub Repository**: Your code must be in a GitHub repository
2. **Streamlit Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Python Dependencies**: All required packages listed in `requirements-deploy.txt`

## 🔧 Deployment Steps

### Step 1: Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Personalized Recommendation System"
   git push origin main
   ```

2. **Verify your repository structure**:
   ```
   personalized_recommendation_system/
   ├── deploy_app.py              # Main deployment app
   ├── requirements-deploy.txt     # Dependencies
   ├── .streamlit/config.toml     # Streamlit config
   ├── src/                       # Source code
   ├── data/                      # Data files
   ├── models/                    # Trained models
   └── README.md                  # Documentation
   ```

### Step 2: Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Configure your app**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `deploy_app.py`
   - **App URL**: Choose a custom URL (optional)

### Step 3: Configure Advanced Settings

1. **Python version**: 3.9 or higher
2. **Requirements file**: `requirements-deploy.txt`
3. **Environment variables**: None required

## 🎯 App Features

Your deployed app will include:

- **📊 Data Overview**: Dataset statistics and visualizations
- **🤖 Model Training**: Train 6 different recommendation models
- **🎯 Recommendations**: Get personalized movie suggestions
- **📈 Model Evaluation**: Compare model performance
- **ℹ️ About**: Project information and documentation

## 🔍 Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all dependencies are in `requirements-deploy.txt`
2. **Path Issues**: Use relative paths in the deployment app
3. **Memory Issues**: The app uses sample data to reduce memory usage
4. **Timeout Issues**: Models are pre-trained to avoid long training times

### Performance Optimizations:

- ✅ Pre-trained models loaded automatically
- ✅ Sample dataset for faster loading
- ✅ Optimized imports and dependencies
- ✅ Error handling for cloud deployment

## 🌐 Access Your Deployed App

Once deployed, your app will be available at:
```
https://your-app-name.streamlit.app
```

## 📊 Model Performance

The deployed system includes:

- **User-based CF**: RMSE ~2.99, MAE ~2.62
- **Item-based CF**: RMSE ~2.97, MAE ~2.58
- **SVD**: RMSE ~2.42, MAE ~1.98
- **NMF**: RMSE ~2.41, MAE ~1.98
- **Content-based**: TF-IDF recommendations
- **Genre-based**: Genre similarity recommendations

## 🎬 Dataset Information

- **10,000 ratings** from MovieLens dataset
- **1,000 users** with rating history
- **500 movies** with metadata
- **7 genres** for content-based filtering

## 📞 Support

If you encounter issues:

1. Check the Streamlit Cloud logs
2. Verify all dependencies are installed
3. Test locally with `streamlit run deploy_app.py`
4. Check the GitHub repository for updates

---

**🎉 Your Personalized Recommendation System is ready for deployment!** 