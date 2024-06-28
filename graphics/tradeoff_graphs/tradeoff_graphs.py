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

# Set up the figure and axes for the grouped bar plots
fig, axs = plt.subplots(3, 1, figsize=(15, 18))

# Define distinct colors for the bars
colors = {'GRU': 'blue', 'LSTM': 'green', 'Simple': 'red'}

# Plotting MSE with distinct colors
mse_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='MSE')
mse_data.plot(kind='bar', ax=axs[0], color=[colors[col] for col in mse_data.columns])
axs[0].set_title('Mean Squared Error (MSE) Comparison')
axs[0].set_xlabel('Model_Target')
axs[0].set_ylabel('MSE')
axs[0].tick_params(axis='x', rotation=90)

# Plotting MAE with distinct colors
mae_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='MAE')
mae_data.plot(kind='bar', ax=axs[1], color=[colors[col] for col in mae_data.columns])
axs[1].set_title('Mean Absolute Error (MAE) Comparison')
axs[1].set_xlabel('Model_Target')
axs[1].set_ylabel('MAE')
axs[1].tick_params(axis='x', rotation=90)

# Plotting R2 with distinct colors
r2_data = combined_data.pivot(index='Model_Target', columns='Model_Type', values='R2')
r2_data.plot(kind='bar', ax=axs[2], color=[colors[col] for col in r2_data.columns])
axs[2].set_title('R-squared (R2) Comparison')
axs[2].set_xlabel('Model_Target')
axs[2].set_ylabel('R2')
axs[2].tick_params(axis='x', rotation=90)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_results_plot_basic.png')