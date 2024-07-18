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
| Random Forest Regressor | Regression      | ISRandomForestRegression         |
| Logistic Regressor      | Classification  | ISLogisticRegression             |
| Linear Regressifier     | Regressification | ISLinearRegressification         |
| Linear Regressor        | Regression      | ISLinearRegression               |

More models are to be added.

Decision Tree Classifier:
-------------------------
Takes a dataset and uses a decision tree classifier to see how well the X columns can predict each private column. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both, you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

.. code-block:: python

    from pandas import pd
    from decision_tree_classifier import DTClassifier

    # Import the combined dataset CSV as a pandas dataframe
    combined_dataset = pd.readcsv('path_to_combined_dataset.csv')

    # Specify the inputs for the classifier
    privatization_type = 'Shuffling'
    RNN_model = 'GRU1'
    target = 'gender'

    # Create decision tree class
    classifier = DTClassifier(privatization_type, RNN_model, target, combined_dataset)

    # Get the best ccp alpha value
    ccp_alpha = classifier.get_best_model(return_model=False)

    # Run the full model - saves graphs, model, and more in the outputs folder
    classifier.run_model(ccp_alpha=ccp_alpha)

Decision Tree Classifier Outputs:
--------------------------------
When the ISDecisionTreeClassifier class is run a variety of outputs can be produced. For outputs to be produced *an output path MUST be declared*.

The recommended organization method is to organize first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

- All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes
- The best ccp alpha fit model
- The y predictions based on the best model
- The classification report from the best model (with an added runtime value)
- The decision tree image from the best model
- Alpha vs accuracy
- Alpha vs graph nodes and graph depth
- Alpha vs total impurity

Note that what items are made and saved can be changed by altering inputs for the functions.

Here are some examples of what the graphs could look like for the basic differential privacy method X GRU 1 layer dimensionality reduction model X target: ethnoracial group. Similar graphics can be produced for other combinations.

.. image:: /docs/graphics/decision_tree_classifier_example/effective_alpha_vs_accuracy.png
   :width: 1080
   :alt: Alpha vs Accuracy
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/effective_alpha_vs_graph_nodes_and_depth.png
   :width: 1080
   :alt: Alpha vs Graph Nodes and Depth
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/effective_alpha_vs_total_impurity.png
   :width: 1080
   :alt: Alpha vs Total Impurity
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/decision_tree_classifier.png
   :width: 1080
   :alt: Decision Tree Example
   :align: center

These SHAP value plots are specific to the Multiracial ethnoracial group. The same type of graphs can be created for all the other ethnoracial groups.

.. image:: /docs/graphics/decision_tree_classifier_example/shap_bar_plot.png
   :width: 1080
   :alt: Decision Tree SHAP Feature Importance
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/shap_bee_swarm_plot.png
   :width: 1080
   :alt: Decision Tree SHAP Bee Swarm Plot
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/shap_heatmap.png
   :width: 1080
   :alt: Decision Tree SHAP Heatmap Plot
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/shap_violin_plot.png
   :width: 1080
   :alt: Decision Tree SHAP Violin Plot
   :align: center

There are also scatter plots for each feature that compare feature value and SHAP value.

.. image:: /docs/graphics/decision_tree_classifier_example/feature_scatter_plots/course_subjects.png
   :width: 1080
   :alt: Course Subject Scatter Plot
   :align: center

.. image:: /docs/graphics/decision_tree_classifier_example/feature_scatter_plots/student_semester.png
   :width: 1080
   :alt: Student Semester Scatter Plot
   :align: center

Decision Tree Alternate:
------------------------
Takes a dataset and uses a decision tree alternate to see how well the X columns can predict future topics as five split columns. This builds off of the alternate preprocessing path for the future topics column that split it into 5 columns. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both, you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

.. code-block:: python

    from pandas import pd
    from decision_tree_classifier import DTClassifier

    # Import the alternate dataset CSV as a pandas dataframe
    alternate_dataset = pd.readcsv('path_to_alternate_dataset.csv')

    # Specify the inputs for the alternate model (regressifier)
    privatization_type = 'Shuffling'
    RNN_model = 'GRU1'
    target = 'future topic 1'

    # Create decision tree class
    alternate = DTAlternate(privatization_type, RNN_model, target, alternate_dataset)

    # Get the best ccp alpha value
    ccp_alpha = alternate.get_best_model(return_model=False)

    # Run the full model - saves graphs, model, and more in the outputs folder
    alternate.run_model(ccp_alpha=ccp_alpha)

Decision Tree Alternate Outputs:
--------------------------------
Contains a variety of outputs from the decision tree alternate function. Organized first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

- All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes
- The best ccp alpha fit model
- The y predictions based on the best model
- The classification report from the best model (with an added runtime value)
- The decision tree image from the best model
- Alpha vs accuracy
- Alpha vs graph nodes and graph depth
- Alpha vs total impurity

Note that what items are made and saved can be changed by altering inputs for the functions.

Decision Tree Regressor:
------------------------
Takes a dataset and uses a decision tree regressor to see how well the X columns can predict each utility column. Specify the privatization method, private column target, and the dimensionality reduction method. The classifier can get the best ccp alpha value (based on maximum x test accuracy) and can also run a single decision tree based on the ccp alpha value. For both, you need to specify how much data you want to read in and then run the test-train split.

The model, various graphs, and the predicted y values are all saved in the outputs folder.

.. code-block:: python

    from pandas import pd
    from decision_tree_regression import DTRegressor

    # Import the combined dataset CSV as a pandas dataframe
    combined_dataset = pd.readcsv('path_to_combined_dataset.csv')

    # Specify the inputs for the regressor
    privatization_type = 'Shuffling'
    RNN_model = 'GRU1'
    target = 'gender'

    # Create decision tree class
    regressor = DTRegressor(privatization_type, RNN_model, target, combined_dataset)

    # Get the best ccp alpha value
    ccp_alpha = regressor.get_best_model(return_model=False)

    # Run the full model - saves graphs, model, and more in the outputs folder
    classifier.run_model(ccp_alpha=ccp_alpha)

Decision Tree Regressor Outputs:
--------------------------------
Contains a variety of outputs from the decision tree regression function. Organized first by privatization type, then by dimensionality reduction type, and then by target. What is saved in each folder:

- All the models (unfitted) from the ccp alpha calculation are saved with some statistics like accuracy, depth, and nodes
- The best ccp alpha fit model
- The y predictions based on the best model
- The metrics CSV which includes: mean squared error, root mean squared error, mean absolute error, median absolute error, r2 score, explained variance score, mean bias deviation, runtime
- The decision tree image from the best model
- Alpha vs accuracy
- Alpha vs graph nodes and graph depth
- Alpha vs total impurity

Note that what items are made and saved can be changed by altering inputs for the functions.
