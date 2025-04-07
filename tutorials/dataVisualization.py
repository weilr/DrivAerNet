# Data Visualization
"""

File: AeroCoefficients_DrivAerNet_FilteredCorrected.csv

This snippet demonstrates data visualization using Seaborn by creating histograms, scatter plots,
and box plots of aerodynamic coefficients from the DrivAerNet dataset.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = '../AeroCoefficients_DrivAerNet_FilteredCorrected.csv'
data = pd.read_csv(file_path)

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(20, 10))

# Histogram of Average Cd
plt.subplot(2, 2, 1)
sns.histplot(data['Average Cd'], kde=True)
plt.title('Histogram of Average Drag Coefficient (Cd)')

# Histogram of Average Cl
plt.subplot(2, 2, 2)
sns.histplot(data['Average Cl'], kde=True)
plt.title('Histogram of Average Lift Coefficient (Cl)')

# Scatter plot of Average Cd vs. Average Cl
plt.subplot(2, 2, 3)
sns.scatterplot(x='Average Cd', y='Average Cl', data=data)
plt.title('Average Drag Coefficient (Cd) vs. Average Lift Coefficient (Cl)')

# Box plot of all aerodynamic coefficients
plt.subplot(2, 2, 4)
melted_data = data.melt(value_vars=['Average Cd', 'Average Cl', 'Average Cl_f', 'Average Cl_r'], var_name='Coefficient',
                        value_name='Value')
sns.boxplot(x='Coefficient', y='Value', data=melted_data)
plt.title('Box Plot of Aerodynamic Coefficients')

plt.tight_layout()
plt.show()

