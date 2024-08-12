"""
Purpose: Project 5 - DOE with rocket engine properties and ML model to replicate CEA
Author: Syam Evani
"""

# Standard imports
import os
import random

# Additional imports
import numpy as np
import itertools
import pandas as pd
from rocketcea.cea_obj import CEA_Obj
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Local imports
# None

#--------------------------------------------------------
# Factor design
#--------------------------------------------------------
# Three factors
factors = ["pc", "mr", "e"]
pc = np.linspace(50, 500, 10)   # [psia] Chamber pressure
mr = np.linspace(1, 10, 10)    # [--]   Ratio of ox/fuel mass
e = np.linspace(10, 100,10)     # [--]   Ratio of exit area to throat area

# Generate all possible combinations
combinations = list(itertools.product(pc, mr, e))

#--------------------------------------------------------
# Generate data and visuals to show the data
#--------------------------------------------------------
results = []
C = CEA_Obj( oxName='O2', fuelName='CH4')
for i, design in enumerate(combinations):
    isp = C.get_Isp(Pc=design[0], MR=design[1], eps=design[2])   # [s] Specific impulse vacuum
    isp = isp + random.randint(-15,15)                           # Add some randomness for artificial test noise
    results.append((design[0], design[1], design[2], isp))

# Convert results to a pandas df and output to text file
isp_df = pd.DataFrame(results, columns=['pc', 'mr', 'e', 'isp'])
txt_file_path = os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project5', 'output', 'isp_df.txt')
with open(txt_file_path, 'w') as file:
    file.write(isp_df.to_string(index=False))

# Make scatter plot of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(isp_df['pc'], isp_df['mr'], isp_df['e'], c=isp_df['isp'], cmap='viridis')
plt.colorbar(sc, label=r'$I_{SP}$ [s]')
ax.set_xlabel('Pc [psia]')
ax.set_ylabel('MR')
ax.set_zlabel(r'$\epsilon$')
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'isp_data.png'))
plt.close()

# Contour plot: Pc vs MR
xi = np.linspace(isp_df['mr'].min(), isp_df['mr'].max(), 100)
yi = np.linspace(isp_df['pc'].min(), isp_df['pc'].max(), 100)
zi = griddata((isp_df['mr'], isp_df['pc']), isp_df['isp'], (xi[None, :], yi[:, None]), method='cubic')
plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(label=r'$I_{SP}$ [s]')
plt.xlabel('MR')
plt.ylabel('Pc [psia]')
plt.title(r'Pc vs MR for $I_{SP}$')
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'contour_pc_mr.png'))
plt.close()

# Contour plot: e vs MR
xi = np.linspace(isp_df['e'].min(), isp_df['e'].max(), 100)
yi = np.linspace(isp_df['mr'].min(), isp_df['mr'].max(), 100)
zi = griddata((isp_df['e'], isp_df['mr']), isp_df['isp'], (xi[None, :], yi[:, None]), method='cubic')
plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(label=r'$I_{SP}$ [s]')
plt.xlabel(r'$\epsilon$')
plt.ylabel('MR')
plt.title(r'MR vs $\epsilon$ of $I_{SP}$')
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'contour_e_mr.png'))
plt.close()

# Generate summary statistics
summary_stats = isp_df.describe(include='all')
print(summary_stats)

#--------------------------------------------------------------------
# Assess the impact of main factors
#--------------------------------------------------------------------
main_effects = {}
for factor in factors:
    main_effects[factor] = isp_df.groupby(factor)['isp'].mean()

# One-Way ANOVA for each factor
factors = ['pc', 'mr', 'e']
for factor in factors:
    groups = [isp_df[isp_df[factor] == level]['isp'] for level in isp_df[factor].unique()]
    f_val, p_val = f_oneway(*groups)
    print(f"ANOVA for {factor}: F-value = {f_val}, p-value = {p_val}")

# Plotting the main effects
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Plot for pc
axs[0].plot(main_effects['pc'].index, main_effects['pc'].values, marker='o')
axs[0].set_title('Main Effect of Pc')
axs[0].set_xlabel('Pc')
axs[0].set_ylabel('Mean ISP')
axs[0].grid()
# Plot for mr
axs[1].plot(main_effects['mr'].index, main_effects['mr'].values, marker='o')
axs[1].set_title('Main Effect of MR')
axs[1].set_xlabel('MR')
axs[1].set_ylabel('Mean ISP')
axs[1].grid()
# Plot for e
axs[2].plot(main_effects['e'].index, main_effects['e'].values, marker='o')
axs[2].set_title('Main Effect of E')
axs[2].set_xlabel(r'$\epsilon$')
axs[2].set_ylabel('Mean ISP')
axs[2].grid()
# Layout
plt.tight_layout()
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'main_effects_subplot.png'))
plt.close()

#------------------------------------------------------------------------
# Assess the impact of interactions
#------------------------------------------------------------------------
# Calculate interaction effects
interaction_effects = {}
for r in range(2, len(factors) + 1):
    for combo in itertools.combinations(factors, r):
        interaction_term = ' x '.join(combo)
        interaction_effects[interaction_term] = isp_df.groupby(list(combo))['isp'].mean()

# Print interaction effects
for key, value in interaction_effects.items():
    print(f"Interaction: {key}")
    print(value)

# Plot heat maps of mean interactions
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Heatmap for pc x mr
interaction = 'pc x mr'
if interaction in interaction_effects:
    heatmap_data = isp_df.pivot_table(values='isp', index='pc', columns='mr', aggfunc='mean')
    sns.heatmap(heatmap_data, ax=axs[0], cmap='viridis')
    axs[0].set_title('Heatmap of Pc and MR Interaction')
    axs[0].set_xlabel('MR')
    axs[0].set_ylabel('Pc')

# Heatmap for pc x e
interaction = 'pc x e'
if interaction in interaction_effects:
    heatmap_data = isp_df.pivot_table(values='isp', index='pc', columns='e', aggfunc='mean')
    sns.heatmap(heatmap_data, ax=axs[1], cmap='viridis')
    axs[1].set_title('Heatmap of Pc and E Interaction')
    axs[1].set_xlabel(r'$\epsilon$')
    axs[1].set_ylabel('Pc')

# Heatmap for mr x e
interaction = 'mr x e'
if interaction in interaction_effects:
    heatmap_data = isp_df.pivot_table(values='isp', index='mr', columns='e', aggfunc='mean')
    sns.heatmap(heatmap_data, ax=axs[2], cmap='viridis')
    axs[2].set_title('Heatmap of MR and E Interaction')
    axs[2].set_xlabel(r'$\epsilon$')
    axs[2].set_ylabel('MR')

# Adjust layout
plt.tight_layout()
plt.savefig(os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project5', 'output', 'interaction_effects_heatmaps.png'))
plt.close()


#--------------------------------------------------------------------
# Slice data into training and test
#--------------------------------------------------------------------
X = isp_df[['pc', 'mr', 'e']].values.tolist()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, isp_df['isp'], test_size=0.2, random_state=42)

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

    # Train a linear regression model
    models["lr"] = LinearRegression().fit(X_train, y_train)
    models["dtr"] = DecisionTreeRegressor().fit(X_train, y_train)
    models["rfr"] = RandomForestRegressor().fit(X_train, y_train)
    models["svr"] = SVR().fit(X_train, y_train)                         # Default rbf kernel, 3 degree polynomial kernel function, uses 1/(n_features) * X.var()) as gamma
    models["knn"]  = KNeighborsRegressor().fit(X_train, y_train)        # By default will use 5 neighbors
    
    # Predict and calculate MSE
    for regressor in models:
        predictions[regressor] = models[regressor].predict(X_test)
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
    plt.title(f"Predictions vs Actual for {regressor} with Design: {min_mse_design} and MSE: {"{:.5f}".format(min_mse)}")
    plt.show()
    #plt.savefig(os.path.join('hw1', 'output', regressor + ".png"))