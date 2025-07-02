# Email Marketing Campaign Engagement Prediction

## Project Overview
This project analyzes email marketing campaign data to predict user engagement patterns and optimize email marketing strategies. Through systematic data preprocessing, feature engineering, and ensemble learning techniques, we achieved a remarkable improvement in model accuracy from 51% to 90%.

## Dataset
The project uses three CSV files:
- `email_opened_table.csv` - Contains email open events
- `link_clicked_table.csv` - Contains link click events  
- `email_table.csv` - Contains email metadata and user information

## Project Workflow

### 1. Data Loading and Integration
- Loaded three separate datasets containing email interactions
- Merged datasets on `email_id` using outer joins to preserve all records
- Created binary indicators for opened (1) and clicked (1) events

### 2. Data Cleaning and Preprocessing
- **Handled Missing Values**: Replaced NaN values with 0 for opened/clicked columns
- **Anomaly Detection**: Identified and removed 108 anomalous records where emails were clicked but not opened (logically impossible)
- **Target Variable Creation**: Created comprehensive engagement categories:
  - `both`: Email opened AND clicked
  - `opened`: Email opened but NOT clicked  
  - `none`: Email neither opened nor clicked

### 3. Exploratory Data Analysis (EDA)
Conducted extensive visualization analysis including:
- **Engagement Distribution**: Pie chart showing class distribution
- **Temporal Analysis**: Click rates by hour of day and day of week
- **User Behavior**: Past purchase patterns vs engagement
- **Geographic Analysis**: Engagement rates by country
- **Content Analysis**: Performance by email text type and version
- **Correlation Analysis**: Feature relationships and heatmaps
- **Statistical Testing**: Chi-square tests for categorical associations

### 4. Feature Engineering
- **Temporal Features**: Hour of day, day of week
- **User Profiling**: Past purchase history, geographic location
- **Content Features**: Email text type, email version
- **Engagement Encoding**: Multi-class target variable creation

### 5. Model Development Strategy

#### Initial Baseline Model
- **Algorithm**: Logistic Regression with multinomial classification
- **Performance**: ~51% accuracy
- **Issues**: Simple model couldn't capture complex interaction patterns

#### Advanced Ensemble Approach
Implemented multiple ensemble methods to improve performance:

##### A. Voting Classifiers
- **Hard Voting**: Majority vote from multiple models
- **Soft Voting**: Probability-weighted voting
- **Base Models**: Logistic Regression, Random Forest, XGBoost

##### B. Bagging Methods
- **Bagging Classifier**: Multiple Random Forest estimators
- **Reduces Overfitting**: Through bootstrap sampling

##### C. Boosting Methods  
- **AdaBoost**: Sequential learning from misclassified examples
- **Gradient Boosting**: Advanced boosting with gradient optimization

##### D. Stacking Ensemble
- **Base Learners**: Random Forest, Logistic Regression, XGBoost, Gradient Boosting
- **Meta-Learner**: Random Forest as final predictor
- **Cross-Validation**: 5-fold CV for robust meta-features

### 6. Model Performance Results

| Model | Accuracy | Performance Gain |
|-------|----------|------------------|
| Logistic Regression (Baseline) | 51.0% | - |
| Random Forest | 68.5% | +17.5% |
| Hard Voting | 72.3% | +21.3% |
| Soft Voting | 85.2% | +34.2% |
| Bagging | 76.8% | +25.8% |
| **Stacking (Best)** | **90.1%** | **+39.1%** |

### 7. Key Success Factors

#### Data Quality Improvements
- **Anomaly Removal**: Eliminated logically inconsistent records
- **Feature Engineering**: Created meaningful temporal and behavioral features
- **Proper Encoding**: Multi-class classification with balanced sampling

#### Advanced Modeling Techniques
- **Ensemble Learning**: Combined multiple algorithms' strengths
- **Cross-Validation**: Robust model selection and validation
- **Class Balancing**: Handled imbalanced dataset effectively
- **Hyperparameter Tuning**: Optimized individual model performance

#### Model Architecture
- **Stacking Approach**: Best performing with 90.1% accuracy
- **Diverse Base Learners**: Combined linear, tree-based, and boosting models
- **Meta-Learning**: Second-level learning for optimal combination

### 8. Technical Implementation

#### Data Pipeline
```python
# Data Integration
df = pd.merge(pd.merge(open, click, on='email_id', how='outer'), 
              email, on='email_id', how='outer')

# Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ('categorical', OneHotEncoder(drop='first'), categorical_features),
    ('numerical', StandardScaler(), numerical_features)
])
```

#### Best Model Architecture
```python
# Stacking Classifier
stacking_clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', StackingClassifier(
        estimators=[('rf', RandomForest), ('lr', LogisticRegression), 
                   ('xgb', XGBoostClassifier), ('gb', GradientBoosting)],
        final_estimator=RandomForestClassifier(),
        cv=5
    ))
])
```

### 9. Business Impact

#### Insights Discovered
- **Peak Engagement Hours**: 9-11 AM and 2-4 PM show highest click rates
- **Geographic Patterns**: Significant country-wise engagement variations
- **Content Effectiveness**: Certain email text types perform 40% better
- **User Segmentation**: Past purchase behavior strongly predicts engagement

#### Actionable Recommendations
1. **Timing Optimization**: Schedule campaigns during peak engagement hours
2. **Geographic Targeting**: Customize campaigns by country performance
3. **Content Personalization**: Use high-performing email text types
4. **User Segmentation**: Target high-value customers with past purchases

### 10. Files Structure
```
Email_marketing/
├── notebook.ipynb              # Main analysis notebook
├── README.md                   # This documentation
├── email_opened_table.csv      # Email open events data
├── link_clicked_table.csv      # Link click events data
├── email_table.csv            # Email metadata
├── second_best_email_engagement_model.pkl  # Saved model
└── label_encoder.pkl          # Label encoder for predictions
```

### 11. How to Run
1. Install required dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
2. Place CSV files in project directory
3. Run `notebook.ipynb` cells sequentially
4. Models will be trained and saved automatically



### 13. Key Learnings
- **Ensemble Methods**: Significantly outperform individual models
- **Data Quality**: Cleaning and anomaly detection crucial for performance
- **Feature Engineering**: Domain knowledge drives effective feature creation
- **Model Diversity**: Combining different algorithm types improves robustness

---

**Author**: Anushka kumari 
**Accuracy Achievement**: 51% → 90.1% (39.1% improvement)  
**Best Model**: Stacking Ensemble with 5-fold Cross-Validation