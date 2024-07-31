# timer.py

import time

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



import pandas as pd

# Define the factors and levels
factors = {
    'n_estimators': ['-', '+'],
    'max_depth': ['-', '+']
}

# Create a full factorial design
def full_factorial_design(factors):
    import itertools
    levels = list(factors.values())
    design = list(itertools.product(*levels))
    return pd.DataFrame(design, columns=factors.keys())

# Generate the design matrix
design_matrix = full_factorial_design(factors)
print(design_matrix)

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Import the time library
t = Timer()
t.start()

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the experiment configurations (2^2 design)
configurations = [
    {'n_estimators': 100, 'max_depth': 2},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 2},
    {'n_estimators': 200, 'max_depth': 10},
]

results = []

# Run the experiments
for config in configurations:
    clf = RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'], random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Store the results
    results.append({
        'n_estimators': config['n_estimators'],
        'max_depth': config['max_depth'],
        'accuracy': accuracy
    })

t.stop()  # A few seconds later

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)

# Display the results
print(results_df)