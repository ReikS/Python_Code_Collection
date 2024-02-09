################################################################
# PROGRAM NAME : Python_Code_Collection.py
# DESCRIPTION : Collection of useful python functions, code snippets, etc.
#
# AUTHOR : ReikS
# CREATION DATE : 2023-19-25
# LAST CHANGE DATE : 2023-19-25
# REVIEWWER : <name of the reviewer>
# REVIEW DATE : <date of the review yyyy-mm-dd>
#	
#	
# SUMMARY : See below a list of the functions and code snippets contained
#           in this collection :
#               000. Coding standards
#               001. Program header template
#               002. Doctrsing template
#               003. Prompt Creator with Bing Copilot
#               004. Python function prompt
#               110. Linear Regression
#               120. Logistic regression
#               130. Regression model comparison
#
#
# REVIEW SUMMARY : <reviewer's notes>
# 
#
# INPUT : none
# 
# 
# OUTPUT : none
#
#
################################################################
# CHANGE TRACKER
# DATE			AUTHOR				DESCRIPTION
# <yyyy-mm-dd>	<name of author>	<short description>
#
################################################################

################################################################
# 000. Coding standards
################################################################

# Adhere to the following coding standards.
# 1. The solution should consist of a single or a small number of scripts.
# 2. Each script has a program header.
# 3. Coding style to be used is functional programming. That means that custom functions are desinged for parts of the solution and then used for the overall solution.
# 4. Each function is commented with a doc string as well as inline comments. The doc string contains a detailed description on the function's arguments and the returned objects. Type hints shall be used.
# 5. PIP-8 applies.

# Use the following templape for the program header:

################################################################
# 001. Program header template
################################################################

################################################################
# PROGRAM NAME : <fill in name of the file>
# DESCRIPTION : <fill in short description>
#
# AUTHOR : <name of the author>
# CREATION DATE : <initial creation of the file in formal yyyy-mm-dd>
# LAST CHANGE DATE : <last change of the file in yyyy-mm-dd>
# REVIEWWER : <name of the reviewer>
# REVIEW DATE : <date of the review yyyy-mm-dd>
# 
# INPUT : <description of input data, files, data sources, links>
#	
#	
# OUTPUT : <description of the scripts output>	
#
#
# SUMMARY : <detailed summary of this program>
# 
# 
#
# REVIEW SUMMARY : <reviewer's notes>
# 
# 
#
################################################################
# CHANGE TRACKER
# DATE			AUTHOR				DESCRIPTION
# <yyyy-mm-dd>	<name of author>	<short description>
#
################################################################


################################################################
# 002. Docstring template
################################################################

def function_name(parameter1, parameter2, ...):
    """
    Function Name:
        function_name

    Description:
        [Brief description of what the function does.]

    Parameters:
        parameter1 (type): [Description of parameter1]
        parameter2 (type): [Description of parameter2]
        ...
        parameterN (type): [Description of parameterN]

    Returns:
        [Description of what the function returns, if applicable.]

    Example:
        [Example of how to use the function.]

    Author:
        [Your Name]

    Date:
        [Date of creation or last modification]

    Notes:
        [Any additional notes or considerations]

    """
    # Function code here


################################################################
# 003. Prompt Creator with Bing Copilot
################################################################

# Hi Bing!
#
# I want you, Bing Copilor, to become my prompt creator. Your goal is to help me create the best possible prompt for my needs. When this is done, the prompt will be used by you, Bing.
# You will follow the following process:
#
# 1. Firstly, you will ask me what the prompt is about. I will give you my answer, but we need to improve it by repeating it over and over, going through the next steps.
#
# 2. Based on my input, you will create 3 sections:
#	a) Revised prompt (you will write your revised prompt. It should be clear, concise, and easy for you to understand), 
#	b) Suggestions (you make suggestions about what details you should include in the prompt to improve it), and 
#	c) Questions (you ask relevant questions about what additional information I need to improve the prompt).
# 3. The prompt you provide should be in the form of a request from me to be executed by you Bing (Copilot).
#
# 4. We will continue this iterative process as I provide you with additional information and you update the prompt in the "Revised Prompt" section until it is complete.

################################################################
# 004. Python function prompt
################################################################

# Context: I am working on a data science project. The goal is to develop a statistical model to be used in credit risk management. I'm working with SQL and python.  

# Objective: My objective is to create well-structured, commented and documented code to accompish that task. I use functional programming as the approach to create the model development code. 

# Task: I need you, Bing, to do the following. I have a pandas dataframe df containing a column y (continuous between 0.0 and 1.0) as target variable for the model and column x being a continuous risk driver (independent variable). Neither x nor y are normally distributed.
#       I want to bin x such that the bins have a good discriminatory power with respect to y - using measures like generalized Gini or Somers'D.
#       Please provide a python function that 
#       a) determines statistically optimal bin for x with respect to y, 
#       b) provide statistical measures to ascertain how good the binning is and 
#       c) Creates a function that can be applied to column x to get a new column x_binned where each value is the midpoint of the bin it belongs to.
#
# The function should adhere to PEP-8 coding standards, use type hints, and be well-documented with comments and the provided docstring format. It should also include a test facility.
#
# Docstring template:
def function_name(parameter1, parameter2, ...):
    """
    Function Name:
        function_name

    Description:
        [Brief description of what the function does.]

    Parameters:
        parameter1 (type): [Description of parameter1]
        parameter2 (type): [Description of parameter2]
        ...
        parameterN (type): [Description of parameterN]

    Returns:
        [Description of what the function returns, if applicable.]

    Example:
        [Example of how to use the function.]

    Author:
        [Your Name]

    Date:
        [Date of creation or last modification]

    Notes:
        [Any additional notes or considerations]

    """
    # Function code here

################################################################
# 110. Linear Regression
################################################################

################################################################
# PROGRAM NAME : linear_regression.py
# DESCRIPTION : Linear regression analysis on the Boston Housing dataset
#
# AUTHOR : Your Name
# CREATION DATE : yyyy-mm-dd
# LAST CHANGE DATE : yyyy-mm-dd
# REVIEWWER : Reviewer's Name
# REVIEW DATE : yyyy-mm-dd
# 
# INPUT : Path to the CSV file containing Boston Housing data
# OUTPUT : Model performance metrics
#
# SUMMARY : This script imports the Boston Housing dataset, performs
# basic analysis, splits the data into training and testing sets,
# fits a linear regression model, scores the model, and evaluates
# model performance.
# 
#
# REVIEW SUMMARY : Reviewer's notes
# 
# 
#
################################################################
# CHANGE TRACKER
# DATE        AUTHOR            DESCRIPTION
# yyyy-mm-dd  Your Name         Initial creation
#
################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, List, Dict

def import_data(filepath: str, delimiter: str = ';') -> pd.DataFrame:
    """
    Function Name:
        import_data

    Description:
        Imports CSV data into a pandas DataFrame.

    Parameters:
        filepath (str): The full path and filename of the CSV file to import.
        delimiter (str): The delimiter used in the CSV file (e.g., ',', ';').

    Returns:
        pd.DataFrame: The imported DataFrame.

    Example:
        df = import_data('path/to/data.csv', ';')

    Author:
        Your Name

    Date:
        yyyy-mm-dd

    Notes:
        Assumes that the first row of the CSV file contains column names.
    """
    # Function code here
    return pd.read_csv(filepath, delimiter=delimiter)

def basic_analysis(df: pd.DataFrame) -> None:
    """
    Function Name:
        basic_analysis

    Description:
        Conducts basic analysis on the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        None

    Example:
        basic_analysis(df)

    Author:
        Your Name

    Date:
        yyyy-mm-dd

    Notes:
        None
    """
    # Function code here
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())

def train_test_split_stratified(df: pd.DataFrame, target: str, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function Name:
        train_test_split_stratified

    Description:
        Performs stratified sampling for train/test split.

    Parameters:
        df (pd.DataFrame): The DataFrame to be split.
        target (str): The target variable for stratification.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.

    Example:
        train, test = train_test_split_stratified(df, 'MEDV', 0.3, 42)

    Author:
        Your Name

    Date:
        yyyy-mm-dd

    Notes:
        Assumes that the target variable is binary.
    """
    # Function code here
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)

    return df_train, df_test


def linear_regression(df_train: pd.DataFrame, df_test: pd.DataFrame, dependent: str, independent: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function Name:
        linear_regression

    Description:
        Fits a linear regression model on the training data and scores on both training and testing data.

    Parameters:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        dependent (str): The dependent variable.
        independent (List[str]): List of independent variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames with predicted values for training and testing.

    Example:
        train_preds, test_preds = linear_regression(train_data, test_data, 'MEDV', ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'])

    Author:
        Your Name

    Date:
        yyyy-mm-dd

    Notes:
        None
    """
    
    # Extract features and target variable
    X_train, y_train = df_train[independent], df_train[dependent]
    X_test, y_test = df_test[independent], df_test[dependent]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)  # Fit the linear regression model

    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return train_preds, test_preds


def model_performance(true_values: pd.Series, predicted_values: pd.Series) -> Dict[str, float]:
    """
    Function Name:
        model_performance

    Description:
        Evaluates the performance of a linear regression model.

    Parameters:
        true_values (pd.Series): True values of the target variable.
        predicted_values (pd.Series): Predicted values of the target variable.

    Returns:
        Dict[str, float]: Dictionary of performance metrics.

    Example:
        metrics = model_performance(df_train['MEDV'], train_preds)

    Author:
        Your Name

    Date:
        yyyy-mm-dd

    Notes:
        None
    """

    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(true_values, predicted_values),
        'Root Mean Squared Error (RMSE)': mean_squared_error(true_values, predicted_values, squared=False),
        'R-squared (R2)': r2_score(true_values, predicted_values)
    }

    return metrics

if __name__ == "__main__":    

    # Import data
    filepath = r'D:\E\Wissensbasis\Projekte\Python_Code_Collection\python_regression_examples\housing.csv'
    delimiter = ';'
    data = import_data(filepath, delimiter)

    # Basic analysis
    basic_analysis(data)

    # Train/test split
    target_variable = 'MEDV'
    
    # Assuming 'data' is your DataFrame and 'target_variable' is 'MEDV'
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)


    # Linear regression
    independent_variables = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    train_preds, test_preds = linear_regression(train_data, test_data, target_variable, independent_variables)

    # Model performance on training data
    train_metrics = model_performance(train_data[target_variable], train_preds)
    print("Training Data Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value}")

    # Model performance on testing data
    test_metrics = model_performance(test_data[target_variable], test_preds)
    print("\nTesting Data Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")


################################################################
# 120. Logistic regression
################################################################

################################################################
# PROGRAM NAME : income_prediction_analysis_enhanced.py
# DESCRIPTION : This script encompasses data exploration, preprocessing, and logistic regression 
#               modeling on the 'adult' dataset, aiming to predict if an individual earns above 50K a year.
#
# AUTHOR : ChatGPT
# CREATION DATE : 2023-10-09
# LAST CHANGE DATE : 2023-10-09
# REVIEWER : [Your Name]
# REVIEW DATE : YYYY-MM-DD
# 
# INPUT : The 'adult' dataset, comprising various demographic and employment-related variables.
#	
# SUMMARY : The script involves the following steps: data loading, basic descriptive statistics generation, 
#           handling missing values, data preprocessing (including one-hot encoding), and logistic regression 
#           modeling, followed by model evaluation.
# 
# REVIEW SUMMARY : [Reviewer's Notes]
# 
################################################################
# CHANGE TRACKER
# DATE			AUTHOR				DESCRIPTION
# 2023-10-09	ChatGPT				Initial version
#
################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset into a pandas DataFrame.

    Args:
    - filepath (str): The path to the dataset.
    
    Returns:
    - DataFrame: The loaded data.
    """
    return pd.read_csv(filepath)

def basic_descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate basic descriptive statistics of the data.

    Args:
    - data (DataFrame): The input data.
    
    Returns:
    - DataFrame: Descriptive statistics of the data.
    """
    return data.describe()

def handle_missing_values(data: pd.DataFrame, strategy: str='drop', columns: list=None) -> pd.DataFrame:
    """
    Handle missing values in the data.

    Args:
    - data (DataFrame): The input data.
    - strategy (str): The strategy to handle missing values ('drop' or 'placeholder'). Default is 'drop'.
    - columns (list of str): The columns to handle missing values. If None, use all columns. Default is None.
    
    Returns:
    - DataFrame: Data after handling missing values.
    """
    if columns is None:
        columns = data.columns
    
    if strategy == 'drop':
        data_clean = data.dropna(subset=columns)
    elif strategy == 'placeholder':
        data_clean = data.copy()
        for col in columns:
            if data[col].dtype == 'object':
                data_clean[col].fillna('Missing', inplace=True)
            else:
                data_clean[col].fillna(0, inplace=True)
    
    return data_clean

def preprocess_data(data: pd.DataFrame, target_var: str) -> tuple:
    """
    Preprocess the data by converting categorical variables into dummy variables.
    
    Args:
    - data (DataFrame): The input data.
    - target_var (str): The target variable.
    
    Returns:
    - tuple: The preprocessed data (X, y) where X contains the input features and y contains the target variable.
    """
    X = pd.get_dummies(data.drop(target_var, axis=1))
    y = data[target_var].apply(lambda x: 1 if x == '>50K' else 0)
    
    return X, y

def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """
    Train a logistic regression model.

    Args:
    - X (DataFrame): The input features.
    - y (Series): The target variable.
    
    Returns:
    - LogisticRegression: The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return model

# Example usage:
# Load data
data_path = 'path_to_your_data/adult.csv'  # Specify your data path
data = load_data(data_path)

# Perform basic descriptive statistics
desc_stats = basic_descriptive_stats(data)

# Handle missing values
data_clean = handle_missing_values(data, strategy='placeholder', columns=['workclass', 'occupation', 'native_country'])

# Preprocess data
X, y = preprocess_data(data_clean, target_var='income')

# Train logistic regression model
model = train_logistic_regression(X, y)


################################################################
# 130. Regression model comparison
################################################################


################################################################
# PROGRAM NAME : linear_regression.py
# DESCRIPTION : Linear regression analysis on the Boston Housing dataset
#
# INPUT : Path to the CSV file containing Boston Housing data
# OUTPUT : Model performance metrics
#
# SUMMARY : This script imports the Boston Housing dataset, performs
# basic analysis, splits the data into training and testing sets,
# fits a linear regression model, scores the model, and evaluates
# model performance.
#
# CHANGE TRACKER
# DATE        AUTHOR            DESCRIPTION
# yyyy-mm-dd  Your Name         Initial creation
#
################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, GammaRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


def import_data(filepath: str, delimiter: str = ';') -> pd.DataFrame:
    """
    Function Name:
        import_data

    Description:
        Imports CSV data into a pandas DataFrame.

    Parameters:
        filepath (str): The full path and filename of the CSV file to import.
        delimiter (str): The delimiter used in the CSV file (e.g., ',', ';').

    Returns:
        pd.DataFrame: The imported DataFrame.

    Example:
        df = import_data('path/to/data.csv', ';')

    Notes:
        Assumes that the first row of the CSV file contains column names.
    """
    return pd.read_csv(filepath, delimiter=delimiter)

def basic_analysis(df: pd.DataFrame) -> None:
    """
    Function Name:
        basic_analysis

    Description:
        Conducts basic analysis on the dataset.

    Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
        None

    Example:
        basic_analysis(df)

    Notes:
        None
    """
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())

def train_test_split_stratified(df: pd.DataFrame, target: str, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function Name:
        train_test_split_stratified

    Description:
        Performs stratified sampling for train/test split.

    Parameters:
        df (pd.DataFrame): The DataFrame to be split.
        target (str): The target variable for stratification.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.

    Example:
        train, test = train_test_split_stratified(df, 'MEDV', 0.3, 42)

    Notes:
        Assumes that the target variable is binary.
    """
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)
    return df_train, df_test

def linear_regression(df_train: pd.DataFrame, df_test: pd.DataFrame, dependent: str, independent: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    Function Name:
        linear_regression

    Description:
        Fits a linear regression model on the training data and scores on both training and testing data.

    Parameters:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        dependent (str): The dependent variable.
        independent (List[str]): List of independent variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, any]: DataFrames with predicted values for training and testing, and the trained model.

    Example:
        train_preds, test_preds, model = linear_regression(train_data, test_data, 'MEDV', ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'])
    """
    
    # Extract features and target variable
    X_train, y_train = df_train[independent], df_train[dependent]
    X_test, y_test = df_test[independent], df_test[dependent]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)  # Fit the linear regression model

    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return train_preds, test_preds, model

def model_performance(true_values: pd.Series, predicted_values: pd.Series) -> Dict[str, float]:
    """
    Function Name:
        model_performance

    Description:
        Evaluates the performance of a linear regression model.

    Parameters:
        true_values (pd.Series): True values of the target variable.
        predicted_values (pd.Series): Predicted values of the target variable.

    Returns:
        Dict[str, float]: Dictionary of performance metrics.

    Example:
        metrics = model_performance(df_train['MEDV'], train_preds)
    """
    metrics = {
        'Mean Absolute Error (MAE)': mean_absolute_error(true_values, predicted_values),
        'Root Mean Squared Error (RMSE)': mean_squared_error(true_values, predicted_values, squared=False),
        'R-squared (R2)': r2_score(true_values, predicted_values)
    }

    return metrics

def generalized_regression(df_train: pd.DataFrame, df_test: pd.DataFrame, dependent: str, independent: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    Function Name:
        generalized_regression

    Description:
        Fits a generalized regression model on the training data and scores on both training and testing data.

    Parameters:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        dependent (str): The dependent variable.
        independent (List[str]): List of independent variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, any]: DataFrames with predicted values for training and testing, and the trained model.

    Example:
        train_preds, test_preds, model = generalized_regression(train_data, test_data, 'MEDV', ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'])
    """
    # Extract features and target variable
    X_train, y_train = df_train[independent], df_train[dependent]
    X_test, y_test = df_test[independent], df_test[dependent]

    # Fit generalized regression model
    model = GammaRegressor()
    model.fit(X_train, y_train)

    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return train_preds, test_preds, model

def random_trees(df_train: pd.DataFrame, df_test: pd.DataFrame, dependent: str, independent: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    Function Name:
        random_trees

    Description:
        Fits a decision tree model on the training data and scores on both training and testing data.

    Parameters:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        dependent (str): The dependent variable.
        independent (List[str]): List of independent variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, any]: DataFrames with predicted values for training and testing, and the trained model.

    Example:
        train_preds, test_preds, model = random_trees(train_data, test_data, 'MEDV', ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'])
    """
    # Extract features and target variable
    X_train, y_train = df_train[independent], df_train[dependent]
    X_test, y_test = df_test[independent], df_test[dependent]

    # Fit decision tree model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return train_preds, test_preds, model

def random_forest(df_train: pd.DataFrame, df_test: pd.DataFrame, dependent: str, independent: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, any]:
    """
    Function Name:
        random_forest

    Description:
        Fits a random forest model on the training data and scores on both training and testing data.

    Parameters:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        dependent (str): The dependent variable.
        independent (List[str]): List of independent variables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, any]: DataFrames with predicted values for training and testing, and the trained model.

    Example:
        train_preds, test_preds, model = random_forest(train_data, test_data, 'MEDV', ['CRIM', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT'])
    """
    # Extract features and target variable
    X_train, y_train = df_train[independent], df_train[dependent]
    X_test, y_test = df_test[independent], df_test[dependent]

    # Fit random forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return train_preds, test_preds, model

def plot_feature_importance(model, features):
    """
    Function Name:
        plot_feature_importance

    Description:
        Plots the feature importance for a given model.

    Parameters:
        model: The trained regression model (e.g., Random Forest).
        features: List of feature names.

    Returns:
        None
    """
    feature_importance = model.feature_importances_
    sorted_idx = feature_importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.show()


if __name__ == "__main__":

    # Import data
    filepath = r'D:\E\Wissensbasis\Projekte\Python_Code_Collection\python_regression_examples\housing.csv'
    delimiter = ';'
    data = import_data(filepath, delimiter)

    # Basic analysis
    basic_analysis(data)

    # Train/test split
    target_variable = 'MEDV'
    
    # Assuming 'data' is your DataFrame and 'target_variable' is 'MEDV'
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # Linear regression
    independent_variables = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    train_preds, test_preds, linear_model = linear_regression(train_data, test_data, target_variable, independent_variables)
    print("Linear Regression Metrics:")
    print("Training Data Metrics:")
    for metric, value in model_performance(train_data[target_variable], train_preds).items():
        print(f"{metric}: {value}")
    print("\nTesting Data Metrics:")
    for metric, value in model_performance(test_data[target_variable], test_preds).items():
        print(f"{metric}: {value}")

    # Generalized regression
    train_preds, test_preds, gen_model = generalized_regression(train_data, test_data, target_variable, independent_variables)
    print("\nGeneralized Regression Metrics:")
    print("Training Data Metrics:")
    for metric, value in model_performance(train_data[target_variable], train_preds).items():
        print(f"{metric}: {value}")
    print("\nTesting Data Metrics:")
    for metric, value in model_performance(test_data[target_variable], test_preds).items():
        print(f"{metric}: {value}")

    # Random Trees
    train_preds, test_preds, tree_model = random_trees(train_data, test_data, target_variable, independent_variables)
    print("\nRandom Trees Metrics:")
    print("Training Data Metrics:")
    for metric, value in model_performance(train_data[target_variable], train_preds).items():
        print(f"{metric}: {value}")
    print("\nTesting Data Metrics:")
    for metric, value in model_performance(test_data[target_variable], test_preds).items():
        print(f"{metric}: {value}")

    # Random Forest
    train_preds, test_preds, forest_model = random_forest(train_data, test_data, target_variable, independent_variables)
    print("\nRandom Forest Metrics:")
    print("Training Data Metrics:")
    for metric, value in model_performance(train_data[target_variable], train_preds).items():
        print(f"{metric}: {value}")
    print("\nTesting Data Metrics:")
    for metric, value in model_performance(test_data[target_variable], test_preds).items():
        print(f"{metric}: {value}")

    # Plot Feature Importance for Random Forest
    plot_feature_importance(forest_model, independent_variables)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, feature_names=independent_variables, filled=True, rounded=True, fontsize=10)
    plt.show()
