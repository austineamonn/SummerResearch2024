Calculating Tradeoffs
=====================

This is where the privacy - utility tradeoff calculations are run on all the different privatization styles. This is done through several model types:

Classification:
---------------
- Calculate the privacy loss on the private columns
- Based on various prebuilt models (ex: sklearn DecisionTreeClassifier)

Regression:
-----------
- Calculate the utility gain on the utility columns
- Based on various prebuilt models (ex: sklearn DecisionTreeRegressor)

Regressification:
-----------------
- Calculate the utility gain on the split 'future topics' columns from alternative future topics preprocessing
- Train a regression model, then convert predictions to integers, then calculate metrics (like accuracy) using classification methods

Each model is represented as a different class. Then a variety of functions can be called depending on the model class. Here is a list of the models:

| Model                   | Model Type      | Class Name                       |
|-------------------------|-----------------|----------------------------------|
| Decision Tree Classifier| Classification  | ISDecisionTreeClassification     |
| Decision Tree Regressifier | Regressification | ISDecisionTreeRegressification   |
| Decision Tree Regressor | Regression      | ISDecisionTreeRegression         |
| Random Forest Classifier | Classification  | ISRandomForestClassification     |
| Random Forest Regressifier | Regressification | ISRandomForestRegressification   |
