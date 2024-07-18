Model Comparison
================

Now that we have various models trained on the various datasets, we need to compare them to decide which dimensionality reduction method was best and which privatization method was best.

Classification Comparison:
---------------------------
This file compares the classification metrics of the various classification machine learning models.

.. code-block:: python

    from classification_comparison import compare_classification_models

    # Call the function to compare models
    compare_classification_models()

Regression Comparison:
-----------------------
This file compares the regression metrics of the various regression machine learning models.

.. code-block:: python

    from regression_comparison import compare_regression_models

    # Call the function to compare models
    compare_regression_models()

Alternate Comparison:
----------------------
This file compares the alternate metrics (same as classification) of the various alternate machine learning models.

.. code-block:: python

    from alternate_comparison import compare_alternate_models

    # Call the function to compare models
    compare_alternate_models()
