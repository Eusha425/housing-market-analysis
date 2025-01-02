# Housing Market Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)

## 📊 Project Overview
A comprehensive machine learning project implementing supervised learning techniques to predict house prices. The project demonstrates the complete machine learning lifecycle, from data preprocessing and feature engineering to model evaluation and optimization. This project was developed as part of the KIT315 unit assessment, demonstrating practical application of machine learning concepts in real-world scenarios.


## 🔍 Key Features
- Implementation of 7 different regression algorithms
- Advanced feature engineering and preprocessing pipeline
- Robust outlier detection and handling using IQR method
- Comprehensive model evaluation and comparison
- Detailed data visualization and analysis
- Hyperparameter optimization using Grid Search CV

## 🛠️ Technologies Used
- Python 3.7+
- Jupyter Notebook
- Key Libraries:
  - pandas & numpy: Data manipulation
  - scikit-learn: Machine learning algorithms
  - seaborn & matplotlib: Data visualization
  - xgboost: Gradient boosting

## 📈 Models Implemented
1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. Kernel Ridge Regression
7. K-Nearest Neighbors Regressor
8. XGBoost Regressor

## 🔄 Project Pipeline

### Data Preprocessing
- Missing value detection and imputation
- Outlier removal using IQR method
- Feature scaling with StandardScaler
- Categorical variable encoding

### Feature Engineering
- Created 'area_per_bedroom' to capture house spaciousness
- Log transformation of 'area' for normal distribution
- One-hot encoding for categorical variables

### Model Development & Evaluation
- Cross-validation with 15 folds
- Grid Search for hyperparameter optimization
- Performance metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²)

## 📊 Results
Ridge Regression emerged as the optimal model, demonstrating:
- Superior R-squared score
- Consistent performance across price ranges
- Robust handling of feature multicollinearity

## 📁 Project Structure
```
housing-market-analysis/
│
├── Housing_Price_Prediction.ipynb   # Main Jupyter notebook
├── README.md                        # Project documentation
└── data/                           
    └── Housing.csv                  # Dataset file
```

## 🚀 Setup and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Eusha425/housing-market-analysis.git
   ```

2. Install required packages:
   ```python
   import pandas as pd
   import numpy as np
   import seaborn as sns
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.preprocessing import StandardScaler
   import xgboost as xgb
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Housing_Price_Prediction.ipynb
   ```

## 📈 Future Improvements
1. **Data Processing**
   - KNN imputation for missing values
   - Advanced outlier detection methods

2. **Feature Engineering**
   - Recursive Feature Elimination (RFE)
   - Domain-specific feature creation

3. **Model Optimization**
   - Bayesian optimization for hyperparameter tuning
   - Ensemble methods exploration

4. **Evaluation**
   - Implementation of MAPE
   - Addition of adjusted R²

## 📚 References
1. Pedregosa et al. (2011) - Scikit-learn: Machine Learning in Python
2. Guyon & Elisseeff (2003) - Variable and Feature Selection
3. Bergstra & Bengio (2012) - Random Search for Hyper-Parameter Optimization
4. Dietterich (2000) - Ensemble Methods in Machine Learning
5. Kohavi (1995) - Cross-Validation and Model Selection

## 🤝 Contributing
Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Eusha425/housing-market-analysis/blob/main/LICENSE) file for details.
