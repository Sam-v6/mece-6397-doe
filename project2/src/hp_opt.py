"""
Purpose: Project 2 - Evaluate different regressors for a 3 factor, 3 level model with different design combos
Author: Syam Evani
"""

# Standard imports
import os
import time

# Additional imports
import pandas as pd
import numpy as np
import itertools
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations

# Local imports
# None

#--------------------------------------------------------------------
# Timing examples
#--------------------------------------------------------------------
# Illustration of a Class Function for Error Exception
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

## Illustration of a Class Function Measure Time Performance
class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        elapsed_time_min = (time.perf_counter() - self._start_time)/60
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        print(f"Elapsed time: {elapsed_time_min:0.4f} minutes")

#--------------------------------------------------------------------
# Design matrix creation
#--------------------------------------------------------------------
# Define the factors and levels

# Provided values (yields accuracy 1.0 for all combos)
# factors = {
#     'n_estimators': [50, 100],
#     'max_depth': [1, 20],
#     'min_samples_split': [2, 6],
#     'min_samples_leaf': [1, 5],
#     'max_samples': [0.5, 0.8]
# }

factors = {
    'n_estimators': [5, 50],
    'max_depth': [1, 10],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 10],
    'max_samples': [0.1, 0.9]
}

# Create a full factorial design
def generate_full_factorial_design(factors):
    levels = list(factors.values())
    design = list(itertools.product(*levels))

    # Convert each combination into a dictionary
    design_matrix = []
    for combo in design:
        single_design = {}
        for i, factor in enumerate(factors.keys()):
            single_design[factor] = combo[i]
        design_matrix.append(single_design)

    # Return list that has dicts inside
    return design_matrix

# Generate the design matrix
design_matrix = generate_full_factorial_design(factors)
# print(design_matrix)

#--------------------------------------------------------------------
# Run experiment
#--------------------------------------------------------------------
# Import the time library
t = Timer()
t.start()

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = []

# Run the experiments
for config in design_matrix:
    clf = RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'], min_samples_split=config['min_samples_split'], min_samples_leaf=config['min_samples_leaf'], bootstrap=True, max_samples=config['max_samples'], random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Store the results
    results.append({
        'n_estimators': config['n_estimators'],
        'max_depth': config['max_depth'],
        'min_samples_split': config['min_samples_split'],
        'min_samples_leaf': config['min_samples_leaf'],
        'max_samples': config['max_samples'],
        'accuracy': accuracy
    })

t.stop()  # A few seconds later

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)

#--------------------------------------------------------------------
# Assess the impact of main factors
#--------------------------------------------------------------------
main_effects = {}
for factor in factors.keys():
    main_effects[factor] = results_df.groupby(factor)['accuracy'].mean()

#------------------------------------------------------------------------
# Assess the impact of interactions
#------------------------------------------------------------------------
# Calculate interaction effects
interaction_effects = {}
factor_names = list(factors.keys())
for r in range(2, len(factor_names) + 1):
    for combo in combinations(factor_names, r):
        interaction_term = ' x '.join(combo)
        interaction_effects[interaction_term] = results_df.groupby(list(combo))['accuracy'].mean().unstack()

#------------------------------------------------------------------------
# Generate Contrast Output
#------------------------------------------------------------------------
contrast_output = pd.DataFrame(columns=['Factor/Interaction', 'Low Level Mean', 'High Level Mean', 'Effect'])
contrast_rows = []

# Calculate contrast for main effects
for factor in factors.keys():
    low_level_mean = main_effects[factor].iloc[0]
    high_level_mean = main_effects[factor].iloc[1]
    effect = high_level_mean - low_level_mean
    contrast_rows.append({
        'Factor/Interaction': factor,
        'Low Level Mean': low_level_mean,
        'High Level Mean': high_level_mean,
        'Effect': effect
    })

# Calculate contrast for interaction effects
for interaction_term, interaction_data in interaction_effects.items():
    for idx, (level, row) in enumerate(interaction_data.iterrows()):
        low_level_mean = row.iloc[0]
        high_level_mean = row.iloc[1]
        effect = high_level_mean - low_level_mean
        contrast_rows.append({
            'Factor/Interaction': f'{interaction_term} (Level {level})',
            'Low Level Mean': low_level_mean,
            'High Level Mean': high_level_mean,
            'Effect': effect
        })

# Save contrast and sort from highest to lowest
contrast_output = pd.DataFrame(contrast_rows)
contrast_output = contrast_output.sort_values(by='Effect', ascending=False)

#------------------------------------------------------------------------
# Print design, main effects contrast, and overall constrast rankings
#------------------------------------------------------------------------
# Print design
print("Full Factorial Design and Percent Reacted:")
print(results_df)

# Print main effects constrast
print("\nMain Effects:")
for factor, effects in main_effects.items():
    print(f"\n{factor}:")
    print(effects)

# Save contrast to text file for interactions
with open(os.path.join(os.getenv('USERPROFILE'),"repos","mece-6397-doe","project2","output","contrast_output.txt"), "w") as file:
    file.write(contrast_output.to_string(index=False))

# Print overall rankings
with open(os.path.join(os.getenv('USERPROFILE'),"repos","mece-6397-doe","project2","output","contrast_output.txt")) as file:
    contents = file.read()
print(contents)

