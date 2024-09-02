import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r'P:\Paris Olmypics total medal tally 2024 - Sheet1.csv')

# Display basic information about the dataset
print(data.head())
print(data.info())
print("Columns in the dataset:", data.columns)

# Data Visualization
plt.figure(figsize=(10, 6))
sns.histplot(data['Total Medals'], kde=True)
plt.title('Distribution of Total Medals')
plt.xlabel('Total Medals')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Gold', y='Silver', data=data)
plt.title('Gold vs Silver Medals')
plt.xlabel('Gold Medals')
plt.ylabel('Silver Medals')
plt.show()

# Prepare data for classification
threshold = 50
data['Medal High'] = (data['Total Medals'] > threshold).astype(int)

# Features and target variable
X = data[['Gold', 'Silver', 'Bronze']]
y = data['Medal High']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Selection - RFE
print("Feature Selection using RFE")
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
rfe = rfe.fit(X_train, y_train)

# Get selected features for RFE
selected_feature_indices_rfe = np.where(rfe.support_)[0]
selected_features_rfe = [X.columns[i] for i in selected_feature_indices_rfe]

print("Selected Features (RFE):", selected_features_rfe)

# Feature Selection - SelectKBest
print("Feature Selection using SelectKBest")
k_best = SelectKBest(score_func=f_classif, k=2)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

# Get selected feature names for SelectKBest
selected_feature_indices_kbest = k_best.get_support(indices=True)
selected_features_kbest = [X.columns[i] for i in selected_feature_indices_kbest]

print("Selected Features (SelectKBest):", selected_features_kbest)

# Train and evaluate classifiers
def evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    print(f"\nEvaluating {model_name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(class_report)

# Classifiers
log_reg = LogisticRegression()
rf_clf = RandomForestClassifier()

# Evaluate without feature selection
print("Classification without Feature Selection")
evaluate_model(X_train, X_test, y_train, y_test, "Logistic Regression", log_reg)
evaluate_model(X_train, X_test, y_train, y_test, "Random Forest Classifier", rf_clf)

# Prepare data with RFE-selected features
X_train_rfe = X_train[selected_features_rfe]
X_test_rfe = X_test[selected_features_rfe]

# Evaluate with RFE-selected features
print("\nClassification with RFE-selected Features")
evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, "Logistic Regression (RFE)", log_reg)
evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test, "Random Forest Classifier (RFE)", rf_clf)

# Prepare data with SelectKBest-selected features
X_train_kbest = pd.DataFrame(X_train_kbest, columns=selected_features_kbest)
X_test_kbest = pd.DataFrame(X_test_kbest, columns=selected_features_kbest)

# Evaluate with SelectKBest-selected features
print("\nClassification with SelectKBest-selected Features")
evaluate_model(X_train_kbest, X_test_kbest, y_train, y_test, "Logistic Regression (SelectKBest)", log_reg)
evaluate_model(X_train_kbest, X_test_kbest, y_train, y_test, "Random Forest Classifier (SelectKBest)", rf_clf)
