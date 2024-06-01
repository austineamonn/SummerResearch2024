import hashlib
import random
import numpy as np
from dictionary import Data
import logging

from config import load_config

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=config["logging"]["level"], format=config["logging"]["format"])

data = Data()
data_dict = data.get_data()
data_gen = data.get_data_generalization()

class Privatizer:
    def __init__(self, config):
        self.config = config
        self.epsilon = self.config["privacy"]["epsilon"]
        self.delta = self.config["privacy"]["delta"]
        self.noise_level = self.config["privacy"]["noise_level"]
        self.method = config["privacy"]["mechanism"]
        self.generalization_level = config["privacy"]["generalization_level"]
        self.parameters = [self.epsilon, self.delta, self.noise_level, self.method, self.generalization_level]

        # Log the structure of data_dict and data_gen for 'gpa'
        logging.debug(f"data_dict structure for gpa: {data_dict.get('gpa')}")
        logging.debug(f"data_gen structure for gpa: {data_gen.get('gpa')}")

    def anonymize_names(self, name):
        return hashlib.sha256(name.encode()).hexdigest()

    def generalize_category(self, category, category_type='race_ethnicity'):
        logging.debug(f"Generalizing category: {category}, Level: {self.generalization_level}, Category Type: {category_type}")
        mapping = data_gen[category_type].get(self.generalization_level, {})
        if isinstance(mapping, dict):
            return mapping.get(category, 'Unknown')
        else:
            logging.error(f"Expected a dict, but got: {type(mapping)}")
            return 'Unknown'

    def apply_generalization(self, value, category_type='student semester'):
        generalization = data_gen[category_type].get(self.generalization_level, 'Unknown')
        logging.debug(f"Generalization function for {category_type}: {generalization}")
        if callable(generalization):
            result = generalization(value)
            logging.debug(f"Generalized value for {category_type}: {result}")
            return result
        else:
            logging.error(f"Generalization for {category_type} is not callable: {generalization}")
            return generalization


    def add_noise_cba(self, value, noise_level=0.1):
        logging.debug(f"Adding CBA noise to value: {value} with noise level: {noise_level}")
        if isinstance(value, (int, float)):
            result = value + random.uniform(-noise_level, noise_level)
            logging.debug(f"Result after adding CBA noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_noise_ddp(self, value, epsilon=1.0):
        logging.debug(f"Adding DDP noise to value: {value} with epsilon: {epsilon}")
        if isinstance(value, (int, float)):
            sensitivity = 1.0
            noise = np.random.laplace(0, sensitivity / epsilon)
            result = value + noise
            logging.debug(f"Result after adding DDP noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_noise_pufferfish(self, value, noise_level=0.1):
        logging.debug(f"Adding Pufferfish noise to value: {value} with noise level: {noise_level}")
        if isinstance(value, (int, float)):
            result = value + random.uniform(-noise_level, noise_level)
            logging.debug(f"Result after adding Pufferfish noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value
    
    def add_random_noise(self, value, noise_level):
        logging.debug(f"Adding random noise to value: {value} with noise level: {noise_level}")
        if isinstance(value, (int, float)):
            result = value + random.uniform(-noise_level, noise_level)
            logging.debug(f"Result after adding random noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_gaussian_noise(self, value, sensitivity, epsilon, delta):
        logging.debug(f"Adding Gaussian noise to value: {value} with sensitivity: {sensitivity}, epsilon: {epsilon}, delta: {delta}")
        if isinstance(value, (int, float)):
            noise = np.random.normal(0, sensitivity / epsilon)
            result = value + noise
            logging.debug(f"Result after adding Gaussian noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_laplace_noise(self, value, sensitivity, epsilon):
        logging.debug(f"Adding Laplace noise to value: {value} with sensitivity: {sensitivity}, epsilon: {epsilon}")
        if isinstance(value, (int, float)):
            noise = np.random.laplace(0, sensitivity / epsilon)
            result = value + noise
            logging.debug(f"Result after adding Laplace noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_exponential_noise(self, value, scale):
        logging.debug(f"Adding Exponential noise to value: {value} with scale: {scale}")
        if isinstance(value, (int, float)):
            noise = np.random.exponential(scale)
            result = value + noise
            logging.debug(f"Result after adding Exponential noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_gamma_noise(self, value, shape, scale):
        logging.debug(f"Adding Gamma noise to value: {value} with shape: {shape}, scale: {scale}")
        if isinstance(value, (int, float)):
            noise = np.random.gamma(shape, scale)
            result = value + noise
            logging.debug(f"Result after adding Gamma noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def add_uniform_noise(self, value, low, high):
        logging.debug(f"Adding Uniform noise to value: {value} with low: {low}, high: {high}")
        if isinstance(value, (int, float)):
            noise = np.random.uniform(low, high)
            result = value + noise
            logging.debug(f"Result after adding Uniform noise: {result}")
            return result
        else:
            logging.error(f"Value is not numerical: {value}")
            return value

    def privatize_dataset(self, dataset):
        dataset['first_name'] = dataset['first_name'].apply(self.anonymize_names)
        logging.debug("First names anonymized.")

        dataset['last_name'] = dataset['last_name'].apply(self.anonymize_names)
        logging.debug("Last names anonymized.")

        dataset['race_ethnicity'] = dataset['race_ethnicity'].apply(lambda x: self.generalize_category(x, category_type='race_ethnicity'))
        logging.debug("Race and Ethnicity generalized.")
        
        dataset['gender'] = dataset['gender'].apply(lambda x: self.generalize_category(x, category_type='gender'))
        logging.debug("Gender generalized.")
        
        dataset['socioeconomic status'] = dataset['socioeconomic status'].apply(lambda x: self.generalize_category(x, category_type='socioeconomic'))
        logging.debug("Socioeconomic status generalized.")
        
        dataset['international'] = dataset['international'].apply(lambda x: self.generalize_category(x, category_type='international'))
        logging.debug("International student status generalized.")
        
        dataset['gpa'] = dataset['gpa'].apply(lambda x: self.apply_generalization(x, category_type='gpa'))
        logging.debug("GPA generalized.")
        dataset['gpa'] = self.noise_addition(dataset, 'gpa')
        logging.debug("GPA generalized with noise addition.")

        logging.debug(f"Student semester before generalization: {dataset['student semester'].head()}")
        dataset['student semester'] = dataset['student semester'].apply(lambda x: self.apply_generalization(x, category_type='student semester'))
        logging.debug(f"Student semester after generalization: {dataset['student semester'].head()}")

        dataset['student semester'] = self.noise_addition(dataset, 'student semester')
        logging.info("Student semester generalized with noise addition.")
        
        print(dataset.head())
        dataset['previous courses count'] = dataset['previous courses count'].apply(lambda x: self.apply_generalization(x, category_type='previous courses count'))
        dataset['previous courses count'] = self.noise_addition(dataset, 'previous courses count')
        logging.debug("Previous Courses generalized with noise addition.")
        
        dataset['subjects of interest'] = dataset['subjects of interest'].apply(lambda x: self.apply_generalization(x, category_type='subjects of interest'))
        logging.debug("Subjects of interest generalized.")
        
        dataset['career aspirations'] = dataset['career aspirations'].apply(lambda x: self.apply_generalization(x, category_type='career aspirations'))
        logging.debug("Career aspirations generalized.")
        
        dataset['extracurricular activities'] = dataset['extracurricular activities'].apply(lambda x: self.apply_generalization(x, category_type='extracurricular activities'))
        logging.debug("extracurricular activities generalized.")

        logging.info("Data anonymization and generalization complete.")
        return dataset
    
    def noise_addition(self, dataset, column):
        try:
            original_length = len(dataset)
            logging.debug(f"Original length of dataset: {original_length}")
            original_column_length = len(dataset[column])
            logging.debug(f"Original length of {column} column: {original_column_length}")

            if self.method == 'Random':
                dataset[column] = dataset[column].apply(lambda x: self.add_random_noise(x, self.noise_level))
            elif self.method == 'Gaussian':
                dataset[column] = dataset[column].apply(lambda x: self.add_gaussian_noise(x, self.sensitivity, self.epsilon, self.delta))
            elif self.method == 'Laplace':
                dataset[column] = dataset[column].apply(lambda x: self.add_laplace_noise(x, self.sensitivity, self.epsilon))
            elif self.method == 'Exponential':
                dataset[column] = dataset[column].apply(lambda x: self.add_exponential_noise(x, scale=1/self.epsilon))
            elif self.method == 'Gamma':
                dataset[column] = dataset[column].apply(lambda x: self.add_gamma_noise(x, shape=2, scale=1))
            elif self.method == 'Uniform':
                dataset[column] = dataset[column].apply(lambda x: self.add_uniform_noise(x, low=-1, high=1))
            elif self.method == 'CBA':
                dataset[column] = dataset[column].apply(lambda x: self.add_noise_cba(x, noise_level=0.1))
            elif self.method == 'DDP':
                dataset[column] = dataset[column].apply(lambda x: self.add_noise_ddp(x, epsilon=1.0))
            elif self.method == 'Pufferfish':
                dataset[column] = dataset[column].apply(lambda x: self.add_noise_pufferfish(x, noise_level=0.1))

            new_length = len(dataset)
            new_column_length = len(dataset[column])
            logging.debug(f"New length of dataset: {new_length}")
            logging.debug(f"New length of {column} column: {new_column_length}")

            if original_length != new_length or original_column_length != new_column_length:
                logging.error("Length of dataset or column changed during noise addition!")
                raise ValueError("Length of dataset or column changed during noise addition!")
            
        except Exception as e:
            logging.error("Error during privatizing gpa: %s", e)
            raise

        logging.debug("Noise added to %s", column)
        return dataset[column]


        
"""
dataset = pd.read_csv("/mnt/data/Dataset.csv")

anonymized_cba_dataset = anonymize_dataset(dataset.copy(), method='cba', generalization_level='full')
anonymized_cba_dataset.to_csv("/mnt/data/anonymized_cba_dataset.csv", index=False)

anonymized_ddp_dataset = anonymize_dataset(dataset.copy(), method='ddp', generalization_level='broad')
anonymized_ddp_dataset.to_csv("/mnt/data/anonymized_ddp_dataset.csv", index=False)

anonymized_pufferfish_dataset = anonymize_dataset(dataset.copy(), method='pufferfish', generalization_level='slight')
anonymized_pufferfish_dataset.to_csv("/mnt/data/anonymized_pufferfish_dataset.csv", index=False)

anonymized_dp_dataset = anonymize_dataset(dataset.copy(), method='dp', generalization_level='full')
anonymized_dp_dataset.to_csv("/mnt/data/anonymized_differential_dataset.csv", index=False)
"""
