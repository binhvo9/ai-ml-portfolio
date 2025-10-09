# 🏡 Vietnam House Price Prediction (Linear Regression)

A simple but clean Machine Learning project to predict housing prices in Vietnam using basic features such as bathrooms, bedrooms, and floors.

## 📊 Dataset
- Source: Crawled from kaggle (Vietnam real estate listings)
- Columns include: Address, Area, Bedrooms, Bathrooms, Floors, Legal status, Furniture state, and Price.

## 🧠 Model
- **Algorithm:** Linear Regression  
- **Features Used:** Bathrooms, Bedrooms, Floors  
- **Target:** Price  

## ⚙️ Workflow
1. Data cleaning and selection  
2. Correlation analysis  
3. Model training with scikit-learn  
4. Evaluation with MAE, RMSE, and R² metrics  

## 📈 Results
| Metric | Score |
|--------|--------|
| MAE | 1.68 |
| RMSE | 2.04 |
| R² | 0.15 |

> The model demonstrates basic predictive power (R² = 0.15) and serves as a foundation for future improvements (e.g., feature engineering, Random Forest, or XGBoost).

## 🚀 How to Run
```bash
python main.py
