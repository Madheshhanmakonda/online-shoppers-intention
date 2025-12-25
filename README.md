# Online Shoppers Purchase Intention Analysis and Prediction

## Project Overview

This project analyzes online user browsing behavior to understand purchasing patterns and predict whether a user session will result in a purchase.  
Using a combination of exploratory data analysis, feature engineering, clustering, and supervised machine learning, the project models customer intent in an e-commerce environment.

The dataset represents real-world session-level data, making the problem relevant to digital marketing, customer analytics, and conversion optimization.

---

## Problem Statement

In e-commerce platforms, only a small fraction of user sessions lead to purchases.  
Identifying high-intent users early can help businesses:

- Improve conversion rates  
- Optimize marketing strategies  
- Personalize user experiences  
- Allocate resources more efficiently  

The objectives of this project are to:
1. Analyze user behavior patterns  
2. Segment users based on engagement  
3. Predict purchase intent using machine learning  

---

## Dataset Description

The dataset contains information about user sessions, including:

- Page visits and durations  
- Bounce and exit rates  
- Traffic source, browser, region, and operating system  
- Visitor type (new vs returning)  
- Weekend indicator  
- Purchase outcome (Revenue)  

The target variable **Revenue** indicates whether a session resulted in a purchase.

---

## Methodology

### 1. Data Preprocessing & Feature Engineering
- Converted categorical variables into numerical representations  
- Applied log transformation to reduce skewness in monetary features  
- Created aggregate behavioral features such as:
  - Total page views  
  - Total session duration  
- Standardized numerical features for clustering and modeling  

### 2. Exploratory Data Analysis (EDA)
- Analyzed class imbalance in purchase outcomes  
- Visualized purchase distribution using count plots and pie charts  
- Studied correlations between behavioral features and revenue  
- Identified patterns indicating high-intent sessions  

### 3. Customer Segmentation (Unsupervised Learning)
- Applied **K-Means clustering** on standardized behavioral features  
- Segmented users into two behavioral groups  
- Visualized clusters using **Principal Component Analysis (PCA)**  

This step helps uncover latent behavioral differences among users beyond the purchase label.

### 4. Predictive Modeling (Supervised Learning)
- Split data using a stratified train–test split  
- Trained a **Random Forest Classifier** with class balancing  
- Evaluated performance using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - ROC-AUC  

### 5. Model Evaluation & Interpretability
- Generated ROC curve and confusion matrix  
- Analyzed feature importance to identify key drivers of purchase intent  
- Observed that engagement-related features strongly influence predictions  

---

## Results

- The model achieved strong predictive performance while handling class imbalance effectively  
- Feature importance analysis showed that session engagement and page value metrics are key indicators of purchase behavior  
- Clustering provided additional insights into distinct browsing patterns among users  

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---

## Project Structure

```
Online-Shoppers-Intention/
├── data/
│   └── online_shoppers_intention.csv
├── notebooks/
│   └── Online_Shoppers_Purchase_Intention.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```
---

## Key Takeaways

- Behavioral analytics can effectively predict purchase intent  
- Combining unsupervised and supervised learning improves understanding of user patterns  
- Proper feature engineering and EDA are crucial for reliable modeling  

---

## Future Improvements

- Experiment with gradient boosting models (XGBoost, LightGBM)  
- Perform hyperparameter optimization  
- Incorporate temporal or session-sequence features  
- Deploy the model as a simple web application or API  

---

## Author

**Madhesh Hanmakonda**  
Final Year Undergraduate Student



