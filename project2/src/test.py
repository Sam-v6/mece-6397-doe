# Define the experiment configurations (2^2 design)
configurations = [
    {'n_estimators': 100, 'max_depth': 2},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 2},
    {'n_estimators': 200, 'max_depth': 10},
]

print(configurations[1]["max_depth"])