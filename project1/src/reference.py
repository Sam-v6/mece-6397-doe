import numpy as np
import pandas as pd

#------------------------------------------------------------------------
## DOE Design Section ##
#------------------------------------------------------------------------
# Define factors and their levels
levels = [-1, 0, 1]  # Assuming -1 is low, 0 is medium, and +1 is high
factor_names = ['Factor A', 'Factor B']

# Generate full factorial design
experiment_design = pd.DataFrame(np.array(np.meshgrid(levels, levels)).T.reshape(-1, 2), columns=factor_names)

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
# Define factors and their levels
levels = [-1, 0, 1]  # Assuming -1 is low, 0 is medium, and +1 is high
factor_names = ['Factor A', 'Factor B']

# Generate full factorial design
experiment_design = pd.DataFrame(np.array(np.meshgrid(levels, levels)).T.reshape(-1, 2), columns=factor_names)

# Simulate experiment outcomes (responses)
# In a real scenario, these would be your observed results from running the experiments
np.random.seed(42)  # For reproducibility
experiment_design['Response'] = np.random.rand(len(experiment_design))

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