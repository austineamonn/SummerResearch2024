import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import os
import textwrap

# Load the dataset
data = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_generation/Dataset.csv')

# Create directory for saving graphs
output_dir = 'data_analysis_graphs'
os.makedirs(output_dir, exist_ok=True)

# Function to wrap text for long labels
def wrap_labels(labels, width):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

# Function to clean up string representations
def clean_label(label):
    return label.strip("[]").replace("'", "")

# Explicitly list string columns
string_columns = [
    'first name', 'last name', 'ethnoracial group', 'gender', 
    'international status', 'socioeconomic status'
]

# 1. Numerical Columns
numerical_columns = data.select_dtypes(include=['float64', 'int64'])
summary_stats_numerical = numerical_columns.describe()
print("Numerical Summary Statistics:")
print(summary_stats_numerical)

for column in numerical_columns.columns:
    if numerical_columns[column].dtype == 'int64':
        # Use bar plot for integer columns
        plt.figure(figsize=(10, 6))
        value_counts = numerical_columns[column].dropna().value_counts().sort_index()
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        for idx, count in enumerate(value_counts):
            plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{column}.png'))
        plt.close()
    else:
        # Use histogram for non-integer columns
        plt.figure(figsize=(10, 6))
        plt.hist(numerical_columns[column].dropna(), bins=30, edgecolor='k')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{column}.png'))
        plt.close()

# 2. String Columns
for column in string_columns:

    plt.figure(figsize=(10, 6))
    top_values = data[column].dropna().value_counts().nlargest(10)
    top_values.plot(kind='bar')
    plt.title(f'Top 10 Value counts for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    for idx, count in enumerate(top_values):
        plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
    plt.xticks(range(len(top_values)), wrap_labels(top_values.index, 20), rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'value_counts_{column}.png'))
    plt.close()

# 3. List Columns
list_columns = [
    'learning style', 'major', 'previous courses', 'course types', 
    'course subjects', 'subjects of interest', 'extracurricular activities', 
    'career aspirations', 'future topics'
]

mlb = MultiLabelBinarizer()
for column in list_columns:
    if column == 'learning style':
        # Special handling for 'learning style'
        def categorize_styles(styles):
            styles = eval(styles)
            if len(styles) > 1:
                return '2 learning styles'
            return styles[0]

        data_exploded = data[column].dropna().apply(categorize_styles)
    else:
        # Safely evaluate the list columns and handle empty lists
        def safe_eval(cell):
            try:
                result = eval(cell)
                return result if isinstance(result, list) else []
            except:
                return []

        data_exploded = data[column].dropna().apply(safe_eval).explode()  # Convert string representation of list to actual list and then explode

    # Remove empty lists and convert lists to strings
    data_exploded = data_exploded[data_exploded.apply(lambda x: x != [])].astype(str)
    data_exploded = data_exploded.apply(clean_label)  # Clean up labels

    plt.figure(figsize=(10, 6))
    top_values = data_exploded.value_counts().nlargest(10)
    top_values.plot(kind='bar')
    plt.title(f'Top 10 Value counts for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    for idx, count in enumerate(top_values):
        plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
    plt.xticks(range(len(top_values)), wrap_labels(top_values.index, 20), rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'value_counts_{column}.png'))
    plt.close()

# Special handling for 'student semester' to keep labels upright
plt.figure(figsize=(10, 6))
value_counts = numerical_columns['student semester'].dropna().value_counts().sort_index()
value_counts.plot(kind='bar')
plt.title('Distribution of student semester')
plt.xlabel('student semester')
plt.ylabel('Frequency')
for idx, count in enumerate(value_counts):
    plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=0, fontsize=8)  # Keep labels upright and smaller font size
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribution_student semester.png'))
plt.close()

# Special handling for 'major' column to exclude 'nan' values
plt.figure(figsize=(10, 6))
major_counts = data['major'].dropna()
major_counts = major_counts[major_counts.apply(lambda x: x != '[]')]
major_counts = major_counts.apply(clean_label)  # Clean up labels
top_values = major_counts.value_counts().nlargest(10)
top_values.plot(kind='bar')
plt.title('Top 10 Value counts for major')
plt.xlabel('major')
plt.ylabel('Count')
for idx, count in enumerate(top_values):
    plt.text(idx, count, str(count), ha='center', va='bottom', fontsize=8)
plt.xticks(range(len(top_values)), wrap_labels(top_values.index, 20), rotation=30, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'value_counts_major.png'))
plt.close()
