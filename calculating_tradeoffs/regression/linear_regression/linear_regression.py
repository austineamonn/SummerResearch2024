import logging
from ast import literal_eval
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
import matplotlib.pyplot as plt
import joblib
import shap
import numpy as np
import time
import os
import pickle

"""
Warning that this class uses a lot of memory and can cause problems if too many iterations of it are called in one run.
"""

class LinearRegressor:
    def __init__(self, privatization_type, RNN_model, target='career aspirations', data=None, output_path=None):
        # Initiate inputs
        self.target = target # Set 'career aspirations' as the target if one is not chosen, other options include: 'future topics'
        self.target_name = target.replace(' ', '_')
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model
        
        # Get Data Paths
        if data is None:
            data_path = f'../../../data_preprocessing/reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_combined.csv'

            self.data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })
        else:
            self.data = data
        self.X_columns = ['gpa', 'student semester', 'learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']

        # Set up Output Paths
        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = f'outputs/{self.privatization_type}/{self.RNN_model}/{self.target_name}'

        # Make the necessary folers
        self.make_folders()

    def make_folders(self, directories=None):
        if directories is None:
            directories = [
                os.path.dirname(f'{self.output_path}/graphs/feature_scatter_plots/example.py'),
                os.path.dirname(f'{self.output_path}/graphs/feature_linear_plots/example.py')
            ]

        for directory in directories:
          # Create the directory if it doesn't exist
          if not os.path.exists(directory):
              os.makedirs(directory, exist_ok=True)

    def split_data(self, full_model=False):
        if full_model:
            self.final_data = self.data
        else:
            self.final_data = self.data.sample(n=2000, random_state=1)
        # Set up X
        self.X = self.final_data[self.X_columns]

        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in self.X_columns:
            self.X.loc[:, column] = self.X[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Fill NaN values with the mean of each column
        self.X = self.X.fillna(self.X.mean())

        # Set up y post stratification
        self.y = self.final_data[[self.target]]

        # Change the lists into just the elements within them if the element is a list otherwise just take the element
        for column in self.y.columns:
            self.y.loc[:, column] = self.y[column].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Test - Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size = 0.8, random_state = 1234)

    def plotter(self, column, model=None, save_fig=False, show_fig=False):
        # Plot the regression using matplotlib
        plt.figure(figsize=(20,10))
        if model is None:
            model = self.model
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7))

        X_col = self.X[[column]]

        ax.scatter(X_col, self.y, color='black')

        ax.plot(X_col, model.predict(self.X), color='red',linewidth=3)
        ax.grid(True,
                axis = 'both',
                zorder = 0,
                linestyle = ':',
                color = 'k')
        ax.tick_params(labelsize = 18)
        ax.set_xlabel('x', fontsize = 24)
        ax.set_ylabel('y', fontsize = 24)

        # Extract single values for formatting
        m_value = self.m[0][0] if isinstance(self.m, np.ndarray) else self.m
        b_value = self.b[0] if isinstance(self.b, np.ndarray) else self.b
        r2_value = self.r2[0] if isinstance(self.r2, np.ndarray) else self.r2


        ax.set_title("Linear Regression Line with Intercept y = {:.2f}x + {:.2f} (R2 = {:.2f})".format(m_value, b_value, r2_value), fontsize = 16 )
        fig.tight_layout()
        if show_fig:
            plt.show()
        if save_fig:
            plt.savefig(f'{self.output_path}/graphs/feature_linear_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)
        else:
            plt.close(fig)

    def run_model(self, model=None, print_results=False, save_files=True, plot_files=True, get_shap=True):
        # Split the data
        self.split_data(full_model=True)

        if model is not None:
            self.model = model
            self.y_pred = self.model.predict(self.X_test)
        else:
            self.model = LinearRegression(fit_intercept=True)

            # Run and time model
            start_time = time.time()
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_test)
            end_time = time.time()
            runtime = end_time - start_time

        # Get Equation of the line
        self.m = self.model.coef_[0][0]
        self.b = self.model.intercept_[0]

        # Get Metrics
        mse = mean_squared_error(self.y_test, self.y_pred, multioutput='raw_values')[0]
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred, multioutput='raw_values')[0]
        medae = median_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred, multioutput='raw_values')[0]
        evs = explained_variance_score(self.y_test, self.y_pred, multioutput='raw_values')[0]
        mbd = np.mean(np.array(self.y_pred) - np.array(self.y_test)) # Mean Bias Deviation

        # Put the resutls into a dataframe
        results = {
            "Slope": [self.m], "Y-intercept": [self.b], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "MedAE": [medae], "R2": [self.r2], "Explained Variance": [evs], "MBD": [mbd], "Runtime": [runtime]
        }
        results_df = pd.DataFrame(results)

        if print_results:
            print(results)

        # Saving to a JSON file
        if save_files:
            results_df.to_csv(f'{self.output_path}/metrics.csv', index=False)

            # Create a new csv file with the predictions
            y_pred_df = self.X_test
            y_pred_df[f'{self.target}'] = self.y_test
            y_pred_df[f'Predicted Class: {self.target}'] = self.y_pred
            y_pred_df.to_csv(f'{self.output_path}/predictions.csv', index=False)

            # Save the model
            self.save_model(f'{self.output_path}/linear_regressor_model.pkl')

        if plot_files:
            # Plot the model
            for name in self.X_columns:
                self.plotter(name, save_fig=True)

        if get_shap:
            # Calculate and plot SHAP values
            self.calculate_shap_values()
            self.plot_shap_values()

        # Ensure all figures were closed
        plt.close('all')

    def calculate_shap_values(self, sample_size=1000, return_values=False):
        # Select a random sample from self.X
        if sample_size < len(self.X):
            self.X_sample = self.X.sample(n=sample_size, random_state=42)
        else:
            self.X_sample = self.X

        explainer = shap.LinearExplainer(self.model, self.X_train)
        self.shap_values = explainer(self.X_sample)  # Obtain SHAP values as an Explanation object

        shap_values_path = f'{self.output_path}/shap_values.npy'
        np.save(shap_values_path, self.shap_values)

        if return_values:
            return self.shap_values

    def plot_shap_values(self, shap_values=None):
        if shap_values is not None:
            self.shap_values = shap_values

        # Generate the SHAP bar plot and capture the axes
        ax = shap.plots.bar(self.shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(20, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'{self.output_path}/graphs/shap_bar_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Scatter plot for each specific feature
        for i, column in enumerate(self.X_columns):
            # Create a figure and axis
            fig, ax = plt.subplots()
            # TODO: Fix the width of learning style graphs to match better
            if column == 'learning style':
                shap.plots.scatter(self.shap_values[:, 2], show=False, ax=ax)
                ax.set_xlim(0.488, 0.51)
            else:
                shap.plots.scatter(self.shap_values[:, i], show=False, ax=ax)

            # Adjust the size
            fig = plt.gcf()
            fig.set_size_inches(15, 6)

            # Save the figure
            plt.savefig(f'{self.output_path}/graphs/feature_scatter_plots/{column}.png', bbox_inches="tight")
            plt.close(fig)

        # Heatmap plot
        ax = shap.plots.heatmap(self.shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(15, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'{self.output_path}/graphs/shap_heatmap.png', bbox_inches="tight")
        plt.close(fig)

        # Bee swarm plot
        ax = shap.plots.beeswarm(self.shap_values, show=False)

        # Get the figure from the axes
        fig = ax.figure

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        fig.savefig(f'{self.output_path}/graphs/shap_bee_swarm_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Violin plot
        fig = plt.figure()
        shap.plots.violin(self.shap_values, show=False)

        # Set the figure size to be wider
        fig = plt.gcf()
        fig.set_size_inches(22, 6)  # Adjust the width and height as needed

        # Save the figure
        plt.savefig(f'{self.output_path}/graphs/shap_violin_plot.png', bbox_inches="tight")
        plt.close(fig)

        # Ensure all figures were closed
        plt.close('all')

    def load_shap_values(self, file_path, return_values=False):
        # Load the .npy file
        self.shap_values = np.load(file_path, allow_pickle=True)

        # Extract the SHAP values, base values, and data
        values = []
        base_values = []
        data = []

        for element in self.shap_values:
            elem_values = []
            elem_data = []
            for item in range(len(element)):
                elem_values.append(element[item].values)
                elem_data.append(element[item].data)
            values.append(elem_values)
            base_values.append(element[0].base_values)
            data.append(elem_data)

        # Convert lists to numpy arrays
        values = np.array(values)
        base_values = np.array(base_values)
        data = np.array(data)

        # Create an Explanation object
        self.shap_values = shap.Explanation(values=values,base_values=base_values, data=data, feature_names=self.X_columns)

        if return_values:
            return self.shap_values

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path, return_file=False):
        try:
            self.model = joblib.load(file_path)
        except Exception as e:
            logging.error(f"Joblib failed to load the model: {e}")
            logging.info("Attempting to load with pickle...")
            try:
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as pickle_e:
                logging.error(f"Pickle also failed to load the model: {pickle_e}")
                raise pickle_e

        if return_file:
            return self.model

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    import cProfile
    import pstats

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    # List of targets for the model 
    targets = ['career aspirations', 'future topics']

    # Get the runtime values for the function
    profiler = cProfile.Profile()
    profiler.enable()

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")

            data_path = f'../../../data_preprocessing/reduced_dimensionality_data/{privatization_type}/{RNN_model}_combined.csv'

            data = pd.read_csv(data_path, converters={
                'learning style': literal_eval,
                'major': literal_eval,
                'previous courses': literal_eval,
                'course types': literal_eval,
                'course subjects': literal_eval,
                'subjects of interest': literal_eval,
                'extracurricular activities': literal_eval,
                'career aspirations': literal_eval,
                'future topics': literal_eval
            })

            for target in targets:
                logging.info(f"Starting {target}")

                target_name = target.replace(' ', '_')

                # Initiate regressor
                regressor = LinearRegressor(privatization_type, RNN_model, target, data=data)
                regressor.run_model(get_shap=False)
                #regressor.split_data(full_model=True)
                #regressor.load_model(f'outputs/{privatization_type}/{RNN_model}/{target_name}/linear_regressor_model.pkl')
                regressor.calculate_shap_values()
                regressor.load_shap_values(f'outputs/{privatization_type}/{RNN_model}/{target_name}/shap_values.npy')
                regressor.plot_shap_values()
                regressor.plotter(save_fig=True)

    # Save the profiling stats to a file
    profile_stats_file = "profile_stats.txt"
    with open(profile_stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()