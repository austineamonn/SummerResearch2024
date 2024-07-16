import pandas as pd
from ast import literal_eval

class AltFutureTopics:
    def __init__(self, privatization_type, RNN_model):
        self.privatization_type = privatization_type
        self.RNN_model = RNN_model

    def get_paths(self):
        self.preprocessed_path = 'Preprocessed_Dataset.csv'

        self.combined_col_path = f'reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_combined.csv'

        self.save_path = f'reduced_dimensionality_data/{self.privatization_type}/{self.RNN_model}_alt_future_topics.csv'

    def int_list_to_separate_cols(self, dataset_take, dataset_add=None, target='future topics', new_name='future topic'):
        # Split the list column into separate columns
        df_expanded = pd.DataFrame(dataset_take[target].tolist(), index=dataset_take.index)

        # Rename the columns
        df_expanded.columns = [f'{new_name} {i+1}' for i in range(df_expanded.shape[1])]

        # If there is no second dataset, set the first dataset to be the second one
        if dataset_add is None:
            dataset_add = dataset_take

        # Concatenate the new columns with the the intended DataFrame
        dataset_final = pd.concat([dataset_add, df_expanded], axis=1).drop(columns=[target])

        return dataset_final

    def get_alt_future_topics(self, paths=None, return_df=True, save_df=False):
        if paths is None:
            self.get_paths()
        else:
            try:
                self.preprocessed_path = paths[0]
                self.combined_col_path = paths[1]
                self.save_path = paths[3]
            except:
                raise ValueError('Wrong paths input format')
            
        # Read in the CSVs
        privatized_df = pd.read_csv(self.preprocessed_path, converters={
                    'future topics': literal_eval
                })
        combined_df = pd.read_csv(self.combined_col_path)
        
        final_df = self.int_list_to_separate_cols(privatized_df, combined_df)

        if save_df:
            final_df.to_csv(self.save_path, index=False)

        if return_df:
            return final_df

# Main execution
if __name__ == "__main__":
    # Import necessary dependencies
    import logging

    # List of RNN models to run
    RNN_model_list = ['GRU1', 'LSTM1', 'Simple1']

    # List of Privatization Types to run 
    privatization_types = ['NoPrivatization', 'Basic_DP', 'Basic_DP_LLC', 'Uniform', 'Uniform_LLC', 'Shuffling', 'Complete_Shuffling']

    for privatization_type in privatization_types:
        logging.info(f"Starting {privatization_type}")
        for RNN_model in RNN_model_list:
            logging.info(f"Starting {RNN_model}")
            # Initiate class instance
            alt_topics_getter = AltFutureTopics(privatization_type, RNN_model)
            alt_topics_getter.get_alt_future_topics(return_df=False, save_df=True)
