import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import ast

# Function to combine lists
def combine_lists(row):
    combined = []
    for col in ['learning style', 'major', 'previous courses', 'course types', 'course subjects', 'subjects of interest', 'extracurricular activities']:
        combined.extend(ast.literal_eval(row[col]))  # Use ast.literal_eval to convert string representation of list to actual list safely
    return [str(item).strip() for item in combined]

# Initialize list to store combined features
transactions = []

# Read the dataset in chunks
chunk_size = 1000
for chunk in pd.read_csv('../../saved_research_files/Dataset.csv', chunksize=chunk_size):
    chunk['combined_features'] = chunk.apply(combine_lists, axis=1)
    transactions.extend(chunk['combined_features'].tolist())

# Encode transactions using sparse matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions, sparse=True)
df_encoded = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

# Apply Apriori algorithm to find frequent itemsets with higher minimum support to reduce memory usage
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Generate association rules with a higher confidence threshold
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display the rules
print(rules.head())
