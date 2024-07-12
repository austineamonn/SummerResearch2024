import pandas as pd
data = pd.read_csv('/Users/austinnicolas/Documents/SummerREU2024/SummerResearch2024/data_preprocessing/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv')
nan_values = data[data['ethnoracial group'].isna()]
print(nan_values)
inf_values = data[data['ethnoracial group'].apply(lambda x: pd.notna(x) and (x == float('inf') or x == float('-inf')))]
print(inf_values)
