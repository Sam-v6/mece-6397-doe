import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import f

#----------------------------------------------------------
# Data collection and analysis
#----------------------------------------------------------
# Assuming you have a dataset with features and target variable (customer satisfaction)
data = pd.read_csv(os.path.join(os.getenv('USERPROFILE'),'repos','mece-6397-doe','project3','ref','Employee Satisfaction Index.csv'))

# Generate summary statistics
summary_stats = data.describe(include='all')
# print(summary_stats)

# Histograms
numerical_columns = ['age', 'salary', 'rating', 'awards']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(numerical_columns):
    axs[i//2, i%2].hist(data[col], bins=10, edgecolor='black')
    axs[i//2, i%2].set_title(f'Histogram of {col.capitalize()}')
    axs[i//2, i%2].set_xlabel(col.capitalize())
    axs[i//2, i%2].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project3', 'output', 'histogram.png'))

# Bar plots
categorical_columns = ['Dept', 'location', 'education', 'recruitment_type']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(categorical_columns):
    data[col].value_counts().plot(kind='bar', ax=axs[i//2, i%2], edgecolor='black')
    axs[i//2, i%2].set_title(f'Bar Plot of {col.capitalize()}')
    axs[i//2, i%2].set_xlabel(col.capitalize())
    axs[i//2, i%2].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project3', 'output', 'bar_plot.png'))

# Box & whisker plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(numerical_columns):
    axs[i//2, i%2].boxplot(data[col])
    axs[i//2, i%2].set_title(f'Box Plot of {col.capitalize()}')
    axs[i//2, i%2].set_xlabel(col.capitalize())
plt.tight_layout()
plt.savefig(os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project3', 'output', 'box_whisker_plot.png'))

#----------------------------------------------------------
# Prepare the dataset
#----------------------------------------------------------
# Converting catgetorial to numberial values for those columns
cleanup_nums = {"Dept":     {"Purchasing": 1, "HR": 2, "Technology":3, "Marketing":4, "Sales":5},
                "location": {"City": 1, "Suburb": 2},
                "education": {"PG":1, "UG":2},
                "recruitment_type": {"On-Campus":1, "Referral":2, "Walk-in":3, "Recruitment Agency":4}}

# Replace the values in the dataframe
data = data.replace(cleanup_nums)

# Assuming the last column is the target variable
X = data.iloc[:, 1:-1]  # features
y = data.iloc[:, -1]    # target

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and transform the data
X_normalized = scaler.fit_transform(X)
#print(X_normalized)

#----------------------------------------------------------
# Data partionining
#----------------------------------------------------------
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Feature Selection using F-test
# Select the top 'k' features, k is set to 10 here, but you can adjust it
selector = SelectKBest(score_func=f_classif, k=12)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

#----------------------------------------------------------
# F-testing
#----------------------------------------------------------
# # Initialize SelectKBest with f_classif
# selector = SelectKBest(score_func=f_classif, k=12)

# # Fit to the training data
# X_new = selector.fit_transform(X, y)

# Get the scores (F-values) computed by f_classif
f_scores = selector.scores_

# Optional: get p-values to understand significance
p_values = selector.pvalues_

# Print the feature names with their corresponding F-scores
for feature, score, p_value in zip(X.columns, f_scores, p_values):
    print(f"Feature: {feature}, F-score: {score:.2f}, P-value: {p_value:.4f}")

#----------------------------------------------------------
# Calculate critical F value
#----------------------------------------------------------
def calculate_critical_f_value(num_groups, total_samples, confidence_level=0.95):
    # Calculate degrees of freedom
    df_between = num_groups - 1  # Degrees of freedom for the numerator (between groups)
    df_within = total_samples - num_groups  # Degrees of freedom for the denominator (within groups)

    # Calculate the critical F-value at the specified confidence level
    critical_f_value = f.ppf(confidence_level, df_between, df_within)
    return critical_f_value

# Example usage:
num_groups = 11
total_samples = 500
confidence_level = 0.95  # 95% confidence

critical_f_value = calculate_critical_f_value(num_groups, total_samples, confidence_level)
print(f"Critical F-value for {num_groups} groups and {total_samples} samples at {confidence_level*100}% confidence level is: {critical_f_value}")

#----------------------------------------------------------
# SVM Classification
#----------------------------------------------------------
# Training the SVM classifier
svm = SVC(kernel='linear')  # Using a linear kernel
svm.fit(X_train_selected, y_train)

# Predicting the test results
y_pred = svm.predict(X_test_selected)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

# Optionally, to see which features were selected:
selected_features_indices = selector.get_support(indices=True)
selected_features_names = X.columns[selected_features_indices]
print(f'Selected features: {selected_features_names}')