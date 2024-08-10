import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import t, ttest_ind

# Assuming you have a dataset with features and target variable (customer satisfaction)
data = pd.read_csv('housing.csv')
# View the dataframe data for Column Names
data.head()

# Converting categorical to numerical values for those columns
cleanup_nums = {"ocean_proximity":     {"<1H OCEAN": 1, "INLAND": 2, "NEAR OCEAN":3, "NEAR BAY":4, "ISLAND":5}}
data = data.replace(cleanup_nums)
data.head(10)

# Adding a satisfied and Unsatisfied column based on whether the medium house value is less than the mean value (== 1) for greater than the mean value (== 0)
# Calculate the mean of the 'medium house value feature' column
mean_score = data['median_house_value'].mean()

# Add a new column 'Below Mean' with 1 if 'Score' is less than the mean, otherwise 0
data['Satisfied: Purchase'] = data['median_house_value'].apply(lambda x: 1 if x < mean_score else 0)

# Step 3: Feature Selection
# Let's say 'price', 'features', and 'quality' are selected features
selected_features = ['housing_median_age', 'total_rooms', 'total_rooms', 'households', 'median_income','ocean_proximity','longitude', 'latitude']

# Step 4: Initial Model Building
X = data[selected_features]
y = data['Satisfied: Purchase']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Integration of t-test
# Let's perform t-test for each feature
for feature in selected_features:
    satisfied = data[data['Satisfied: Purchase'] == 1][feature]
    dissatisfied = data[data['Satisfied: Purchase'] == 0][feature]
    
    # Calculate t-test statistic and p-value
    t_stat, p_value = ttest_ind(satisfied, dissatisfied)
    
    # Calculate critical t-value from t-distribution
    n1 = len(satisfied)
    n2 = len(dissatisfied)
    dof = n1 + n2 - 2  # Degrees of freedom for independent two-sample t-test
    critical_t = t.ppf(0.05, dof)  # Using 0.05 significance level
    
    print(f"T-test results for '{feature}': t-statistic={t_stat}, p-value={p_value}, critical t-value={critical_t}")

# Step 6: Feature Adjustment (if necessary)
# Let's say we decide to keep all features for simplicity

# Step 7: Model Building and Evaluation
# Let's build a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 8: Deployment and Monitoring
# Deployment steps would depend on your production environment