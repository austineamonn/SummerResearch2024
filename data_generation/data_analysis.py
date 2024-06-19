import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import os
import sys
import textwrap

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DataAnalysis:
    def __init__(self, config, data):
        # Output Directory
        self.output_dir = config["data_analysis"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        # Dataset
        self.data = data
        self.string_cols = config["data_analysis"]["string_cols"]
        self.list_cols = config["data_analysis"]["list_cols"]
        self.total_cols = self.string_cols + self.list_cols
    
    def wrap_labels(self, labels, width):
        return ['\n'.join(textwrap.wrap(label, width)) for label in labels]
    
    def clean_label(self, label):
        return label.strip("[]").replace("'", "")
    
    def analyze_numerical_columns(self):
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64'])
        summary_stats_numerical = numerical_columns.describe()
        
        summary_stats_numerical.to_csv(os.path.join(self.output_dir, 'numerical_summary_statistics.csv'))
        
        for column in numerical_columns.columns:
            plt.figure(figsize=(10, 6))
            if numerical_columns[column].dtype == 'int64':
                value_counts = numerical_columns[column].dropna().value_counts().sort_index()
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                for idx, count in enumerate(value_counts):
                    plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
            else:
                plt.hist(numerical_columns[column].dropna(), bins=30, edgecolor='k')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'distribution_{column}.png'))
            plt.close()
    
    def analyze_columns(self):
        mlb = MultiLabelBinarizer()
        for column in self.total_cols:
            if column in self.list_cols:
                if column == 'learning style':
                    def categorize_styles(styles):
                        styles = eval(styles)
                        if len(styles) > 1:
                            return '2 learning styles'
                        return styles[0]

                    data_exploded = self.data[column].dropna().apply(categorize_styles)
                else:
                    def safe_eval(cell):
                        try:
                            result = eval(cell)
                            return result if isinstance(result, list) else []
                        except:
                            return []

                    data_exploded = self.data[column].dropna().apply(safe_eval).explode()
            else:
                data_exploded = self.data[column].dropna()

            # Remove empty lists and convert lists to strings if it's a list column
            if column in self.list_cols:
                data_exploded = data_exploded[data_exploded.apply(lambda x: x != [])].astype(str)
                data_exploded = data_exploded.apply(self.clean_label)

            plt.figure(figsize=(10, 6))
            top_values = data_exploded.value_counts().nlargest(10)
            top_values.plot(kind='bar')
            plt.title(f'Top 10 Value counts for {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            for idx, count in enumerate(top_values):
                plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
            plt.xticks(range(len(top_values)), self.wrap_labels(top_values.index, 20), rotation=30, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'value_counts_{column}.png'))
            plt.close()
    
    def analyze_special_columns(self):
        # Special handling for 'student semester'
        plt.figure(figsize=(10, 6))
        value_counts = self.data['student semester'].dropna().value_counts().sort_index()
        value_counts.plot(kind='bar')
        plt.title('Distribution of student semester')
        plt.xlabel('student semester')
        plt.ylabel('Frequency')
        for idx, count in enumerate(value_counts):
            plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distribution_student semester.png'))
        plt.close()
        
        # Special handling for 'major'
        plt.figure(figsize=(10, 6))
        major_counts = self.data['major'].dropna()
        major_counts = major_counts[major_counts.apply(lambda x: x != '[]')]
        major_counts = major_counts.apply(self.clean_label)
        top_values = major_counts.value_counts().nlargest(10)
        top_values.plot(kind='bar')
        plt.title('Top 10 Value counts for major')
        plt.xlabel('major')
        plt.ylabel('Count')
        for idx, count in enumerate(top_values):
            plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
        plt.xticks(range(len(top_values)), self.wrap_labels(top_values.index, 20), rotation=30, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'value_counts_major.png'))
        plt.close()

    def calculate_missing_percentage(self):
        missing_percentage = self.data.isnull().mean() * 100
        missing_percentage_df = missing_percentage.reset_index()
        missing_percentage_df.columns = ['Column', 'Missing_Percentage']
        
        missing_percentage_df.to_csv(os.path.join(self.output_dir, 'missing_percentage.csv'), index=False)

    def analyze_data(self):
        self.analyze_numerical_columns()
        self.analyze_columns()
        self.analyze_special_columns()
        self.calculate_missing_percentage()

if __name__ == "__main__":
    # Import necessary dependencies
    from config import load_config

    # Load configuration and data
    config = load_config()
    data_path = config["running_model"]["data path"]
    data = pd.read_csv(data_path)

    analyzer = DataAnalysis(config, data)
    analyzer.analyze_data()