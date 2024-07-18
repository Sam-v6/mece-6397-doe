"""
Purpose: HW1 - Evaluate different regressors for a 3 factor, 3 level model with different design combos
Author: Syam Evani
"""

# Standard imports
import os

# Additional imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from pyDOE3 import fullfact
import matplotlib.pyplot as plt 

# Local imports
# None

#--------------------------------------------------------------------
# Create design combos
#--------------------------------------------------------------------
# Define the levels for each factor (transformation)
# Features X1, X2, X3 -- three of them
# 0: No transformation, 1: Logarithmic, 2: Square root, 3: Square
levels = {0: lambda x: x,
          1: lambda x: np.log(x + 0.1),
          2: np.sqrt,
          3: np.square}

# Generating a full factorial design for 3 factors, each with 3 levels
design = fullfact([3, 3, 3])

# Show the design matrix
# print(design)
print(f"Number of Design:",len(design))

#--------------------------------------------------------------------
# Create sample dataset
#--------------------------------------------------------------------
# Sample dataset
# Replace this with your actual dataset
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2*X[:, 0] + 3*np.log(X[:, 1] + 1) + np.sqrt(X[:, 2])  # Sample target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--------------------------------------------------------------------
# Apply different regression approaches
#--------------------------------------------------------------------
# Placeholder for results and best transformed features
results = {"lr": [],        # Linear regression
           "dtr": [],       # Decision tree regression
           "rfr": [],       # Random forest regression
           "svr": [],       # Support vector regression
           "knn": []        # K-nearest neighbors regression
            }
models = {}
predictions = {}
best_transformed_features = None

for i in range(len(design)):
    # Apply transformations based on the design matrix
    X_train_transformed = np.column_stack([levels[design[i, j]](X_train[:, j]) for j in range(X.shape[1])])
    X_test_transformed = np.column_stack([levels[design[i, j]](X_test[:, j]) for j in range(X.shape[1])])
    
    # Train a linear regression model
    models["lr"] = LinearRegression().fit(X_train_transformed, y_train)
    models["dtr"] = DecisionTreeRegressor().fit(X_train_transformed, y_train)
    models["rfr"] = RandomForestRegressor().fit(X_train_transformed, y_train)
    models["svr"] = SVR().fit(X_train_transformed, y_train)                         # Default rbf kernel, 3 degree polynomial kernel function, uses 1/(n_features) * X.var()) as gamma
    models["knn"]  = KNeighborsRegressor().fit(X_train_transformed, y_train)        # By default will use 5 neighbors
    
    # Predict and calculate MSE
    for regressor in models:
        predictions[regressor] = models[regressor].predict(X_test_transformed)
        mse = mean_squared_error(y_test, predictions[regressor])
    
        # Store the results
        results[regressor].append((design[i], mse, predictions[regressor]))

#--------------------------------------------------------------------
# Post process and plot different regressors for comparison
#--------------------------------------------------------------------
 # Plotting predictions against actual values
plt.figure(figsize=(10, 6))

# Post-process different regression approaches
for regressor in results:
    # Find the design with the lowest MSE
    min_mse_design, min_mse, best_predictions = min(results[regressor], key=lambda x: x[1])

    # Print the design, MSE, and feature values with the lowest error
    print(f"Regressor: {regressor}")
    print(f"Design with lowest MSE: {min_mse_design}, MSE: {min_mse}")

    plt.scatter(y_test, best_predictions, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title(f"Predictions vs Actual for {regressor} with Design: {min_mse_design} and MSE: {"{:.5f}".format(mse)}")
    plt.savefig(os.path.join('hw1', 'output', regressor + ".png"))