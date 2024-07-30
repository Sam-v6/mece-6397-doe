"""
Purpose: Project 1 - Driver script for factorial design and main effects calculation
Author: Syam Evani
"""

# Standard imports
import os

# Additional imports
import pandas as pd
import numpy as np
from itertools import combinations

# Local imports
# None

#------------------------------------------------------------------------
# DOE Design Section
#------------------------------------------------------------------------
# Define factors and their levels
levels = [0, 1]  # Assuming -1 is low, 0 is medium, and +1 is high
factors={
        'Feed Rate': {"Min": 10, "Max": 15},
        'Catalyst':{"Min": 1, "Max": 2},
        'Stir Rate':{"Min": 100, "Max": 120},
        'Temperature':{"Min": 140, "Max": 180}, 
        'Concentration':{"Min": 3, "Max": 6}
        }

# Generate full factorial design
experiment_design = pd.DataFrame(np.array(np.meshgrid(levels, levels, levels, levels, levels)).T.reshape(-1, 5), columns=factors.keys())

# Capture what pattern was generated
pattern_list = []
for i in range(0,32):
    pattern = []
    for factor in experiment_design.columns:
        pattern.append(experiment_design[factor][i])
    pattern_list.append(pattern)

# Map the levels to their corresponding min and max values
for factor in experiment_design.columns:
    for i in range(0,32):
        if experiment_design[factor][i] == 0:
            experiment_design[factor][i] = factors[factor]["Min"]
        else:
            experiment_design[factor][i] = factors[factor]["Max"]

# Add the pattern and Percent Reacted to the design
experiment_design['Pattern'] = pattern_list

# Display the design with Percent Reacteds
np.random.seed(42)  # For reproducibility
experiment_design['Percent Reacted'] = np.random.rand(len(experiment_design))*100

#------------------------------------------------------------------------
# DOE Main Effects Calculation
#------------------------------------------------------------------------
main_effects = {}
for factor in factors.keys():
    main_effects[factor] = experiment_design.groupby(factor)['Percent Reacted'].mean()

#------------------------------------------------------------------------
# DOE Interaction Effects Calculation
#------------------------------------------------------------------------
# Calculate interaction effects
interaction_effects = {}
factor_names = list(factors.keys())
for r in range(2, len(factor_names) + 1):
    for combo in combinations(factor_names, r):
        interaction_term = ' x '.join(combo)
        interaction_effects[interaction_term] = experiment_design.groupby(list(combo))['Percent Reacted'].mean().unstack()

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
print(experiment_design)

# Print main effects constrast
print("\nMain Effects:")
for factor, effects in main_effects.items():
    print(f"\n{factor}:")
    print(effects)

# Save contrast to text file for interactions
with open(os.path.join(os.getenv('USERPROFILE'),"repos","mece-6397-doe","project1","output","contrast_output.txt"), "w") as file:
    file.write(contrast_output.to_string(index=False))

# Print overall rankings
with open(os.path.join(os.getenv('USERPROFILE'),"repos","mece-6397-doe","project1","output","contrast_output.txt")) as file:
    contents = file.read()

print(contents)

