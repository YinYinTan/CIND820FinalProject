# CIND820 Final Project: Classification Models for Predicting the Mental Health of Health Care Workers

## Link to Dataset
Statistic Canada's "Impacts of COVID-19 on Health Care Workers: Infection Prevention and Control, Public Use Microdata File", containing the original dataset and associated documentation: https://www150.statcan.gc.ca/n1/pub/13-25-0004/132500042021001-eng.htm
* **The original dataset is also uploaded with its original filename "HS.csv".**

## Overview of Project
This project has 3 goals:
1.	Determine the dominant predictors of the mental health status of health care workers (knowledge discovery). Initial exploratory data analysis will help reveal what variables are correlated with mental health status.
2.	Utilize several supervised machine learning classification approaches using Python programming language to actually predict the mental health of individual health care workers. Data pre-processing (e.g. one-hot encoding), feature selection (filter, wrapper), and hyperparameter tuning will be used. 
3.	Identify the best classification model from the ones developed. Model evaluation techniques (e.g. train-test split, k-folds cross validation) and evaluation measures (e.g. accuracy, precision, recall, F1-score, AUC, brier score) will be compared for each model.

3 types of clasification models were built:
1. Random Forest
2. Support Vector Machine (SVM)
3. XGBoost

Each type of classification model was applied to 2 datasets:
1. The original working dataset
2. A working dataset with principal components to reduce dimensionality.

7 Jupyter (ipynb) notebooks were created:
* One notebook for initial data transformation and exploratory data analysis (EDA) of the raw dataset.
* Six notebooks for each combination of classifier type (i.e. Random Forest, SVM, and XGBoost) and dataset (i.e. original features, principal components).


## 1_Data_Transformation_&_EDA.ipynb
This file contains the initial data transformation and exploratory data analysis (EDA) of the raw dataset. The file is split into multiple sections:
* **1. Preliminary Exploratory Data Analysis (EDA)**: Creation of the pandas profile of the raw dataset.
* **2. Data Transformation**
* **3. EDA**
  * **3.1 Univariate Analysis**: Creation of the pandas profile of the transformed dataset.
  * **3.2 Bivariate Analysis: Correlation Analysis**: Analysis of correlations between variables, both against the target variable and also each other.
* **4. Development of Final Working Dataset A - Original Features**: Creation of the working dataset.
  * **4.1 Initial Feature Selection**: Initial analysis and removal of features from the dataset.
  * **4.2 One-Hot Key Encoding**: Applied to nominal categorical variables.
  * **4.3 Save Working Dataset A**
* **5. Development of Final Working Dataset B - Principal Component Analysis (PCA)**: Transformation of the working dataset with principal components to reduce dimensionality.

There are 4 outputs from this file (saved in the same GitHub folder):
* **1_Raw Dataset Profile.html**: The pandas profile of the raw dataset.
* **2. Transformed Dataset Profile.html**: The pandas profile of the transformed dataset.
* **3a. Working Dataset.csv**: The working dataset.
* **3b. Working Dataset - PCA.csv**: The working dataset with principal components to reduce dimensionality.

## 4_Random_Forest_Classification_Models.ipynb
This file contains several random forest classification models, using the working dataset obtained from '1_Data_Transformation_&_EDA.ipynb' (i.e. 3a. Working Dataset.csv). The file is split into multiple sections:
* **1. All Features in Working Dataset - Evaluation using Train-Test Split**: A random forest classification model, using package defaults, and no feature selection.
* **2. All Features in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the random forest classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Features in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4A. Selected Features - Filter Method**: Identification of the most important features, testing to see best number of features to use, then finally an evaluation of the model using repeated 10-fold cross validation. The model uses the best combination of parameters previously identified during hyperparameter tuning.
  * **4A.1 Top Features Ranked by Importance**: Visualizing the importance of features.
  * **4A.2 Top 1 to 20 Features - Filter Method**: Investigation of the best number of features to use by obtaining Recall scores using 1 to 20 features.
  * **4A.3 Top 11 Features, Filter Method - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the best number of features previously identified, using repeated 10-fold cross validation.
* **4B. 7 Selected Features using Recursive Feature Elimination (RFE) - Wrapper Method**: Evaluation of a random forect classification model that uses RFE for feature selection. Initially, the best number of features previously identified was used (11), along with best combination of parameters previously identified during hyperparameter tuning. However, computational time was significant, and the model had to be adjusted. Final evaluation was performed using repeated 5-fold cross validation.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, 4A, and 4B.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 3 to 15 folds.

## 4a_Random_Forest_Classification_Models_PCA.ipynb
This file contains several random forest classification models, using the working dataset with principal components obtained from '1_Data_Transformation_&_EDA.ipynb' (i.e. 3b. Working Dataset - PCA.csv). The file is split into multiple sections:
* **1. All Principal Components in Working Dataset - Evaluation using Train-Test Split**: A random forest classification model, using package defaults, and all principal components.
* **2. All Principal Components in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the random forest classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Principal Components in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4. Top Principal Components**: Building a model using a select number of principal components.
  * **4.1 Top 1 to 20 Principal Components**: Investigation of the number of principal components to use by obtaining Recall scores using 1 to 20 principal components.
  * **4.2 Top 17 Principal Components - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the ideal number of principal components previously identified, using repeated 10-fold cross validation and the best combination of parameters previously identified during hyperparameter tuning.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, and 4.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 3 to 15 folds.

## 5_SVM_Classification_Models.ipynb
This file contains several SVM classification models, using the working dataset obtained from '1_Data_Transformation_&_EDA.ipynb' (i.e. 3a. Working Dataset.csv). The file is split into multiple sections:
* **1. All Features in Working Dataset - Evaluation using Train-Test Split**: A SVM classification model, using package defaults, and no feature selection.
* **2. All Features in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the SVM classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Features in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4. Selected Features - Filter Method**: Building a model using a select number of features.
  * **4.1 Top 1 to 40 Features - Filter Method**: Investigation of the best number of features to use by obtaining Recall scores using 1 to 20 features.
  * **4.2 Top 25 Features, Filter Method - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the best number of features previously identified, using repeated 10-fold cross validation and the best combination of parameters previously identified during hyperparameter tuning.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, 4A, and 4B.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 5 to 12 folds.

## 5a_SVM_Classification_Models_PCA.ipynb
This file contains several SVM classification models, using the working dataset with principal components obtained from '1_Data_Transformation_&_EDA.ipynb' (i.e. 3b. Working Dataset - PCA.csv). The file is split into multiple sections:
* **1. All Principal Components in Working Dataset - Evaluation using Train-Test Split**: A SVM classification model, using package defaults, and all principal components.
* **2. All Principal Components in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the SVM classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Principal Components in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4. Top Principal Components**: Building a model using a select number of principal components.
  * **4.1 Top 1 to All Principal Components**: Investigation of the number of principal components to use by obtaining Recall scores using 1 to all principal components.
  * **4.2 Top 24 Principal Components - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the ideal number of principal components previously identified, using repeated 10-fold cross validation and the best combination of parameters previously identified during hyperparameter tuning.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, and 4.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 5 to 12 folds.

## 6a_XGBoost_Classification_Models_PCA.ipynb
This file contains several XGBoost classification models, using the working dataset with principal components obtained from '1_Data_Transformation_&_EDA.ipynb' (i.e. 3b. Working Dataset - PCA.csv). The file is split into multiple sections:
* **1. All Principal Components in Working Dataset - Evaluation using Train-Test Split**: A XGBoost classification model, using package defaults, and all principal components.
* **2. All Principal Components in Working Dataset - Hyperparameter Tuning**: Identification of best combination of parameters for the XGBoost classification model.
  * **2.1 Random Search with Cross Validation**: A random search to narrow down possible parameter values.
  * **2.2 Grid Search with Cross Validation**: A grid search to test multiple combination of parameter values, chosen based on the results of the random search.
* **3. All Principal Components in Working Dataset, Using Parameters from Grid Search - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model using best combination of parameters previously identified in section 2.
* **4. Top Principal Components**: Building a model using a select number of principal components.
  * **4.1 Top 1 to All Principal Components**: Investigation of the number of principal components to use by obtaining Recall scores using 1 to all principal components.
  * **4.2 Top 22 Principal Components - Evaluation using Repeated 10-fold Cross Validation**: Evaluation of the model with the ideal number of principal components previously identified, using repeated 10-fold cross validation and the best combination of parameters previously identified during hyperparameter tuning.
* **5. Comparison of Models**: Identification of the best model from the final ones developed in sections 1, 3, and 4.
* **6. Stability of Best Model by Varying k-folds for Cross Validation**: Investigation of stability of the best model by obtaining Recall scores from cross validation using 5 to 12 folds.

## A. Combined Technical Reports
HTML file of the 7 Jupyter notebooks combined.

## Files Still in Progress
  * 6_XGBoost_Classification_Models.ipynb
