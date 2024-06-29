import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
file_path_gru = '../../calculating_tradeoffs/tradeoff_results_basic/model_evaluation_results_GRU1.csv'
file_path_lstm = '../../calculating_tradeoffs/tradeoff_results_basic/model_evaluation_results_LSTM1.csv'
file_path_simple = '../../calculating_tradeoffs/tradeoff_results_basic/model_evaluation_results_Simple1.csv'

data_gru = pd.read_csv(file_path_gru)
data_lstm = pd.read_csv(file_path_lstm)
data_simple = pd.read_csv(file_path_simple)

# Add a column to each dataframe to indicate the model type
data_gru['Model_Type'] = 'GRU'
data_lstm['Model_Type'] = 'LSTM'
data_simple['Model_Type'] = 'Simple'

# Combine the dataframes
combined_data = pd.concat([data_gru, data_lstm, data_simple], ignore_index=True)

# Normalize the Model_Target values for consistency
combined_data['Model_Target'] = combined_data['Model_Target'].str.replace(' ', '').str.replace('_', ' ').str.title()

# Function to extract the relevant category part from Model_Target
def extract_target_category(model_target):
    parts = model_target.split()
    return ' '.join(parts[1:])

# Apply the function to extract the target category
combined_data['Target_Category'] = combined_data['Model_Target'].apply(extract_target_category)

# Sort the data by Target_Category and Model_Target for proper grouping
combined_data.sort_values(by=['Target_Category', 'Model_Target'], inplace=True)

# Verify the sorted data
print("Sorted Data:")
print(combined_data[['Model_Target', 'Target_Category']].head(20))

# Pivot the data for plotting
mse_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='MSE')
mae_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='MAE')
r2_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='R2')

# Ensure the order of indices is maintained in pivot tables
mse_data = mse_data.loc[combined_data['Model_Target'].unique()]
mae_data = mae_data.loc[combined_data['Model_Target'].unique()]
r2_data = r2_data.loc[combined_data['Model_Target'].unique()]

# Function to add category separators
def add_separators(ax, data, combined_data):
    prev_category = None
    for i, target in enumerate(data.index):
        category = combined_data.loc[combined_data['Model_Target'] == target, 'Target_Category'].values[0]
        if category != prev_category:
            ax.axvline(i - 0.5, color='black', linewidth=1.5)
            prev_category = category

# Set up the figure and axes for the grouped bar plots with separators
fig, axs = plt.subplots(3, 1, figsize=(15, 18))

# Define distinct colors for the bars
colors = {'GRU': 'blue', 'LSTM': 'green', 'Simple': 'red'}

# Plotting MSE with distinct colors and separators
mse_data.plot(kind='bar', ax=axs[0], color=[colors[col] for col in mse_data.columns])
axs[0].set_title('Mean Squared Error (MSE) Comparison')
axs[0].set_xlabel('Model_Target')
axs[0].set_ylabel('MSE')
axs[0].tick_params(axis='x', rotation=90)
add_separators(axs[0], mse_data, combined_data)

# Plotting MAE with distinct colors and separators
mae_data.plot(kind='bar', ax=axs[1], color=[colors[col] for col in mae_data.columns])
axs[1].set_title('Mean Absolute Error (MAE) Comparison')
axs[1].set_xlabel('Model_Target')
axs[1].set_ylabel('MAE')
axs[1].tick_params(axis='x', rotation=90)
add_separators(axs[1], mae_data, combined_data)

# Plotting R2 with distinct colors and separators
r2_data.plot(kind='bar', ax=axs[2], color=[colors[col] for col in r2_data.columns])
axs[2].set_title('R-squared (R2) Comparison')
axs[2].set_xlabel('Model_Target')
axs[2].set_ylabel('R2')
axs[2].tick_params(axis='x', rotation=90)
add_separators(axs[2], r2_data, combined_data)

# Adjust layout and show plot
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_results_plot_basic2.png')

plt.close()

"""
Do a neural network for the private columns
"""