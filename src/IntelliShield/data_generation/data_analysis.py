import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import os
import sys
import textwrap
import seaborn as sns

# Add the SummerResearch2024 directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DataAnalysis:
    def __init__(self, data, output_dir: str, string_cols: list = None, list_cols: list = None, numerical_cols: str = None):
        # Output Directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Dataset
        self.data = data
        if string_cols is None:
            self.string_cols = ['first name','last name','ethnoracial group','gender','international status','socioeconomic status']
        else:
            self.string_cols = string_cols
        if list_cols is None:
            self.list_cols = ['learning style','major','previous courses','course types','course subjects','subjects of interest','extracurricular activities','career aspirations','future topics']
        else:
            self.list_cols = list_cols
        self.total_cols = self.string_cols + self.list_cols
        if numerical_cols is None:
            self.numerical_cols = ['gpa', 'student semester']
        else:
            self.numerical_cols = numerical_cols
    
    def wrap_labels(self, labels, width):
        return ['\n'.join(textwrap.wrap(label, width)) for label in labels]
    
    def clean_label(self, label):
        return label.strip("[]").replace("'", "")
    
    def analyze_numerical_cols(self):
        summary_stats_numerical = self.numerical_cols.describe()
        
        summary_stats_numerical.to_csv(os.path.join(self.output_dir, 'numerical_summary_statistics.csv'))
        
        for column in self.numerical_cols.columns:
            plt.figure(figsize=(10, 6))
            if self.numerical_cols[column].dtype == 'int64':
                value_counts = self.numerical_cols[column].dropna().value_counts().sort_index()
                value_counts.plot(kind='bar')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                for idx, count in enumerate(value_counts):
                    plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
            else:
                plt.hist(self.numerical_cols[column].dropna(), bins=30, edgecolor='k')
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

    # Eventually change this to be run on the lower dimensional stuff (RNNs)
    def correlation_analysis(self):
        # Convert categorical columns to numerical using one-hot encoding
        data_encoded = pd.get_dummies(self.data, drop_first=True)
        
        # Compute the correlation matrix
        correlation_matrix = data_encoded.corr()
        
        # Visualize the correlation matrix using a heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'))
        plt.close()

    def outlier_detection(self):
        for column in self.numerical_cols.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.numerical_cols[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'boxplot_{column}.png'))
            plt.close()

    def analyze_data(self):
        self.analyze_numerical_cols()
        self.analyze_columns()
        self.analyze_special_columns()
        self.calculate_missing_percentage()
        #self.correlation_analysis()
        self.outlier_detection()

if __name__ == "__main__":
    # Load configuration and data
    data_path = 'Dataset.csv'
    data = pd.read_csv(data_path)

    analyzer = DataAnalysis(data)
    analyzer.analyze_data()