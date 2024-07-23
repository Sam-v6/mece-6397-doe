"""
Purpose: Project 1 - Driver script for factorial design and main effects calculation
Author: Syam Evani
"""

# Standard imports
import os

# Additional imports
import pandas as pd
import numpy as np
from pyDOE3 import fullfact
import matplotlib.pyplot as plt 

# Local imports
# None

#------------------------------------------------------------------------
## DOE Design Section ##
#------------------------------------------------------------------------
# Define factors and their levels
levels = [0, 1]  # Assuming -1 is low, 0 is medium, and +1 is high
factors={
        'Feed Rate': {"Min": 0, "Max": 10},
        'Catalyst':{"Min": 0, "Max": 10},
        'Stir Rate':{"Min": 0, "Max": 10},
        'Temperature':{"Min": 0, "Max": 10}, 
        'Concentration':{"Min": 0, "Max": 10}
        }

# Generate full factorial design
experiment_design = pd.DataFrame(np.array(np.meshgrid(levels, levels)).T.reshape(1, 5), columns=factors.keys())

# Simulate experiment outcomes (responses)
# In a real scenario, these would be your observed results from running the experiments
np.random.seed(42)  # For reproducibility
experiment_design['Response'] = np.random.rand(len(experiment_design))

## DOE Main Effects Calculation Section ##
main_effects = {}
for factor in factor_names:
    main_effects[factor] = experiment_design.groupby(factor)['Response'].mean()

# Display the design and the calculated main effects
print("Full Factorial Design and Responses:")
print(experiment_design)
print("\nMain Effects:")
for factor, effects in main_effects.items():
    print(f"\n{factor}:")
    print(effects)

#------------------------------------------------------------------------
## DOE Secondary Effects Calculation Section ##
#------------------------------------------------------------------------
# Calculate main effects
main_effects = {}
for factor in factor_names:
    main_effects[factor] = experiment_design.groupby(factor)['Response'].mean()

# Calculate interaction effects
# Add an interaction term to the DataFrame
experiment_design['Interaction AxB'] = experiment_design['Factor A'] * experiment_design['Factor B']
interaction_effects = experiment_design.groupby(['Factor A', 'Factor B'])['Response'].mean().unstack()

# Display the design and the calculated effects
print("Full Factorial Design and Responses:")
print(experiment_design)
print("\nMain Effects:")
for factor, effects in main_effects.items():
    print(f"\n{factor}:")
    print(effects)

print("\nInteraction Effects (AxB):")
print(interaction_effects)