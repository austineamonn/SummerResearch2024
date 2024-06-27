import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = '../../calculating_tradeoffs/tradeoff_results_basic/model_evaluation_results_GRU1.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Function to simplify and categorize model names
def simplify_model_name(row):
    if '_private_columns' in row:
        return row.replace('_private_columns', ''), 'Private Columns'
    elif '_utility_colums' in row:
        return row.replace('_utility_colums', ''), 'Utility Columns'
    elif '_all_columns' in row:
        return row.replace('_all_columns', ''), 'All Columns'
    else:
        return row, 'Unknown'

# Apply the function to the data
data[['Model', 'Category']] = data['Model_Target'].apply(simplify_model_name).apply(pd.Series)

# Sort the data by Model and Category
data = data.sort_values(by=['Model', 'Category'])

# Set the figure size for better readability
plt.figure(figsize=(14, 12))

# Plotting MSE
plt.subplot(3, 1, 1)
for category, group in data.groupby('Category'):
    plt.bar(group['Model'] + ' (' + group['Category'] + ')', group['MSE'], label=category)
plt.ylabel('MSE')
plt.title('Mean Squared Error (MSE) by Model')
plt.xticks([])  # Remove x-ticks for the first plot
plt.legend()

# Plotting MAE
plt.subplot(3, 1, 2)
for category, group in data.groupby('Category'):
    plt.bar(group['Model'] + ' (' + group['Category'] + ')', group['MAE'], label=category)
plt.ylabel('MAE')
plt.title('Mean Absolute Error (MAE) by Model')
plt.xticks([])  # Remove x-ticks for the second plot
plt.legend()

# Plotting R2
plt.subplot(3, 1, 3)
for category, group in data.groupby('Category'):
    plt.bar(group['Model'] + ' (' + group['Category'] + ')', group['R2'], label=category)
plt.xlabel('Model_Target')
plt.ylabel('R2')
plt.title('R-squared (R2) by Model')
plt.xticks(rotation=45, ha='right')  # Rotate x-ticks for the third plot
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_results_plot.png')