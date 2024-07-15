from sklearn.tree import DecisionTreeClassifier
from ast import literal_eval
import pandas as pd
from sklearn.model_selection import train_test_split
import shap

data_path = '../../data_preprocessing/reduced_dimensionality_data/NoPrivatization/GRU1_combined.csv'

data = pd.read_csv(data_path, nrows=1000, converters={
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

 # Set up X
X = data[['learning style', 'major', 'previous courses', 'course types', 
        'course subjects', 'subjects of interest', 'extracurricular activities']]
# Change the lists into just the elements within them
for column in X.columns:
    X.loc[:,column] = X[column].apply(lambda x: x[0])

# Set up y
y = data[['ethnoracial group']]

# Change the doubles into integers
y = y.astype(int)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X)

# Create cohorts based on the 'ethnoracial group' feature
ethnoracial_labels = [
    'European American or white', 
    'Latino/a/x American', 
    'African American or Black', 
    'Asian American', 
    'Multiracial', 
    'American Indian or Alaska Native', 
    'Pacific Islander'
]

ethnoracial_groups = [
    ethnoracial_labels[int(data['ethnoracial group'].iloc[i])] 
    for i in range(data.shape[0])
]
cohort_explanation = shap.Explanation(values=shap_values.values, base_values=shap_values.base_values, 
                                      data=shap_values.data, feature_names=shap_values.feature_names)
cohorts = cohort_explanation.cohorts(ethnoracial_groups)

# Visualize SHAP values for each cohort
shap.plots.bar(cohorts.abs.mean(0))