# CIND820 Final Project: Classification Models for Predicting the Mental Health of Health Care Workers

Link to Statistic Canada's "Impacts of COVID-19 on Health Care Workers: Infection Prevention and Control, Public Use Microdata File", containing the original dataset and associated documentation: https://www150.statcan.gc.ca/n1/pub/13-25-0004/132500042021001-eng.htm

## 1_Data_Transformation_&_EDA.ipynb
This file contains the initial data transformation and exploratory data analysis (EDA) of the raw dataset. There are 4 outputs from this file:
* **1. Raw Dataset Profile.html**: The pandas profile of the raw dataset.
* **2. Transformed Dataset Profile.html**: The pandas profile of the transformed dataset.
* **3a. Working Dataset.csv**: The working dataset.
* **3b. Working Dataset - PCA.csv**: The working dataset with principal components to reduce dimensionality.

## 4_Random_Forest_Classification_Models.ipynb
This file contains several random forest classification models. The file is split into multiple sections:
* **1. All Features in Working Dataset - Evaluation using Train-Test Split**: A random forest classification model, using package defaults, and no feature selection.
* **2. All Features in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the random forest classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Features in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4A. Selected Features - Filter Method**: Identification of the most important features, testing to see best number of features to use, then finally an evaluation of the model using repeated 10-fold cross validation. The model uses the best combination of parameters previously identified during hyperparameter tuning.
  * **4A.1 Top 20 Features Ranked by Importance**: Visualizing the importance of the top 20 features.
  * **4A.2 Top 1 to 10 Features - Filter Method**: Investigation of the best number of features to use by obtaining Recall scores using 1 to 10 features.
  * **4A.3 Top 5 Features, Filter Method - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the best number of features previously identified, using repeated 10-fold cross validation.
* **4B. 5 Selected Features using Recursive Feature Elimination (RFE) - Wrapper Method**: Evaluation of a random forect classification model that uses RFE for feature selection, using the best number of features previously identified and the best combination of parameters previously identified during hyperparameter tuning. Evaluation is performed using repeated 10-fold cross validation.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, 4A, and 4B.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 3 to 15 folds.
