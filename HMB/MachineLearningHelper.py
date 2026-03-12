import os, optuna, pickle, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def GetScalerObject(scalerName):
  r'''
  Retrieve a scikit-learn scaler object based on the specified scaler name.

  This function returns an instance of a scaler from scikit-learn's preprocessing module
  according to the provided scaler name. Supported scalers include StandardScaler,
  MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, and Normalizer.

  Supported scalers include:
    - "Standard": Standard Scaler (Standardization)
    - "MinMax": Min-Max Scaler (Normalization)
    - "Robust": Robust Scaler (Robust to outliers)
    - "MaxAbs": Max Absolute Scaler (Scales each feature by its maximum absolute value)
    - "QT": Quantile Transformer (Transforms features to follow a uniform or normal distribution)
    - "Normalizer": Max normalizer (Scales samples individually to unit norm)
    - "L1": L1 Normalizer (Scales samples to have L1 norm equal to 1)
    - "L2": L2 Normalizer (Scales samples to have L2 norm equal to 1)

  Parameters:
    scalerName (str): Name of the scaler to retrieve.
      Supported values: "Standard", "MinMax", "Robust", "MaxAbs", "QT", "Normalizer", "L1", "L2".

  Returns:
    object: An instance of the requested scaler class from scikit-learn.

  Raises:
    ValueError: If an invalid scaler name is provided.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.MachineLearningHelper as mlh
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset.
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=0.2,
      random_state=np.random.randint(0, 10000),
    )
    # Get a scaler object (e.g., Standard Scaler).
    scaler = mlh.GetScalerObject("Standard")
    # Fit the scaler on the training data and transform both training and testing data.
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    # Initialize and train a Logistic Regression model.
    model = mlh.GetMLClassificationModelObject("LR")
    model.fit(xTrainScaled, yTrain)
    # Make predictions on the testing data.
    yPred = model.predict(xTestScaled)
    # Calculate and print the accuracy of the model.
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")
  '''

  if (scalerName is None):
    return None

  # Support common aliases and full class names
  if (scalerName in ["Standard", "StandardScaler"]):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler()
  elif (scalerName in ["MinMax", "MinMaxScaler"]):
    from sklearn.preprocessing import MinMaxScaler
    return MinMaxScaler()
  elif (scalerName in ["Robust", "RobustScaler"]):
    from sklearn.preprocessing import RobustScaler
    return RobustScaler()
  elif (scalerName in ["MaxAbs", "MaxAbsScaler"]):
    from sklearn.preprocessing import MaxAbsScaler
    return MaxAbsScaler()
  elif (scalerName in ["QT", "QuantileTransformer"]):
    from sklearn.preprocessing import QuantileTransformer
    return QuantileTransformer()
  elif (scalerName == "Normalizer"):
    from sklearn.preprocessing import Normalizer
    return Normalizer(norm="max", copy=True)
  elif (scalerName == "L1"):
    from sklearn.preprocessing import Normalizer
    return Normalizer(norm="l1", copy=True)
  elif (scalerName == "L2"):
    from sklearn.preprocessing import Normalizer
    return Normalizer(norm="l2", copy=True)
  # You can add more scalers as needed.
  else:
    raise ValueError(f"Invalid scaler name: {scalerName}.")


def ListScikitMachineLearningClassifiers():
  r'''
  List all available machine learning classifiers in scikit-learn.

  This function retrieves all classifier estimators from scikit-learn's all_estimators function
  and returns them as a dictionary containing the classifier name, class object, module name,
  class name, and import statement for each classifier.

  Returns:
    dict: Dictionary where keys are classifier names and values are dictionaries with class object, module name, class name, and import statement.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    classifiers = mlh.ListScikitMachineLearningClassifiers()
    for name, info in classifiers.items():
      print(f"Classifier Name: {name}")
      print(f"Class Object: {info['class']}")
      print(f"Module: {info['module']}")
      print(f"Class Name: {info['name']}")
      print(f"Import Statement: {info['import']}")
      print("-" * 40)
  '''

  # Import the all_estimators function from sklearn.utils module.
  from sklearn.utils import all_estimators

  # Retrieve all classifier estimators using type filter for classifiers.
  classifiers = all_estimators(type_filter="classifier")

  # Create a dictionary to store classifier names and their class objects.
  classifiersDict = {}
  # Iterate through the classifiers and store them in a dictionary.
  for name, cls in classifiers:
    moduleList = cls.__module__.split(".")
    # Remove the intermediate modules that begin with "_".
    moduleList = [m for m in moduleList if not m.startswith("_")]
    # Join the remaining modules to form the full module path.
    moduleName = ".".join(moduleList)
    # Store the classifier class object in a dictionary with its name as the key.
    classifiersDict[name] = {
      "class" : cls,
      "module": moduleName,
      "name"  : cls.__name__,
      "import": f"from {moduleName} import {cls.__name__}",
    }

  # Return the dictionary containing classifier names and their class objects.
  return classifiersDict


def ListScikitMachineLearningRegressors():
  r'''
  List all available machine learning regressors in scikit-learn.

  This function retrieves all regressor estimators from scikit-learn's all_estimators function
  and returns them as a dictionary containing the regressor name, class object, module name,
  class name, and import statement for each regressor.

  Returns:
    dict: Dictionary where keys are regressor names and values are dictionaries with class object, module name, class name, and import statement.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    
    regressors = mlh.ListScikitMachineLearningRegressors()
    
    for name, info in regressors.items():
      print(f"Regressor Name: {name}")
      print(f"Class Object: {info['class']}")
      print(f"Module: {info['module']}")
      print(f"Class Name: {info['name']}")
      print(f"Import Statement: {info['import']}")
      print("-" * 40)
  '''

  from sklearn.utils import all_estimators

  regressors = all_estimators(type_filter="regressor")
  regressorsDict = {}
  for name, cls in regressors:
    moduleList = cls.__module__.split(".")
    moduleList = [m for m in moduleList if not m.startswith("_")]
    moduleName = ".".join(moduleList)
    regressorsDict[name] = {
      "class" : cls,
      "module": moduleName,
      "name"  : cls.__name__,
      "import": f"from {moduleName} import {cls.__name__}",
    }
  return regressorsDict


def GetFilteredClassifiers(clsList=[]):
  r'''
  Retrieve a filtered list of scikit-learn classifier classes.

  This function returns a list of classifier classes from scikit-learn, excluding those that require
  special arguments (such as "base_estimator" or "estimators") or are known to fail with certain inputs.
  If a specific list of classifier names is provided, only those classifiers are returned.

  Parameters:
    clsList (list, optional): List of classifier names to include. If empty or None, all available classifiers are considered.

  Returns:
    list: List of tuples containing classifier names and their corresponding class objects.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    # Get all filtered classifiers.
    classifiers = mlh.GetFilteredClassifiers()
    for name, cls in classifiers:
      print(f"Classifier Name: {name}, Class: {cls}")
    # Get only specific classifiers.
    classifiers = mlh.GetFilteredClassifiers(["RandomForestClassifier", "LogisticRegression"])
    for name, cls in classifiers:
      print(f"Classifier Name: {name}, Class: {cls}")
  '''

  from sklearn.utils import all_estimators

  # If no specific classifiers are provided, retrieve all classifiers from scikit-learn.
  if ((clsList is None) or (len(clsList) <= 0)):
    # Get all available classifiers in scikit-learn.
    estimators = all_estimators(type_filter="classifier")
    # Select only the classifiers that don't have the arguments:
    # "base_estimator" and "estimators".
    args = ["base_estimator", "estimators"]

    # Classifier: CategoricalNB failed with error: Negative values in data passed to CategoricalNB (input X)
    # Classifier: ComplementNB failed with error: Negative values in data passed to ComplementNB (input X)
    # Classifier: MultiOutputClassifier failed with error: __init__() missing 1 required positional argument: 'estimator'
    # Classifier: MultinomialNB failed with error: Negative values in data passed to MultinomialNB (input X)
    # Classifier: NuSVC failed with error: specified nu is infeasible
    # Classifier: OneVsOneClassifier failed with error: __init__() missing 1 required positional argument: 'estimator'
    # Classifier: OneVsRestClassifier failed with error: __init__() missing 1 required positional argument: 'estimator'
    # Classifier: OutputCodeClassifier failed with error: __init__() missing 1 required positional argument: 'estimator'
    # Classifier: RadiusNeighborsClassifier failed with error: No neighbors found for test samples array(...),
    #   you can try using larger radius, giving a label for outliers, or considering removing them from your dataset.
    # Classifier: StackingClassifier failed with error: __init__() missing 1 required positional argument: 'estimators'
    # Classifier: VotingClassifier failed with error: __init__() missing 1 required positional argument: 'estimators
    # Classifier: ClassifierChain failed with error: __init__() missing 1 required positional argument: 'base_estimator'

    toIgnore = [
      est
      for est in estimators
      for varName in est[1].__init__.__code__.co_varnames
      if (varName in args)
    ]

    # Classifier: ComplementNB failed with error: Negative values in data passed to ComplementNB (input X)
    toIgnore = toIgnore + [
      "CategoricalNB",
      "ComplementNB",
      "MultiOutputClassifier",
      "MultinomialNB",
      "NuSVC",
      "OneVsOneClassifier",
      "OneVsRestClassifier",
      "OutputCodeClassifier",
      "RadiusNeighborsClassifier",
      "StackingClassifier",
      "VotingClassifier",
      "ClassifierChain",
    ]

    estimators = [
      est
      for est in estimators
      if (est[0] not in toIgnore)
    ]

    for i in range(len(estimators)):
      # Print the classifier name and class and function arguments.
      print(f"{i + 1}: {estimators[i][0]}", flush=True)

  else:
    # If a specific list of classifiers is provided, filter the classifiers based on the provided list.
    estimators = []
    for cls in clsList:
      estimators.append(
        [
          cls,
          [
            est for est in all_estimators(type_filter="classifier")
            if (est[0] == cls)
          ][0][1]
        ]
      )

  # Return the list of classifiers.
  return estimators


def GetFilteredRegressors(regList=[]):
  r'''
  Retrieve a filtered list of scikit-learn regressor classes.

  This function returns a list of regressor classes from scikit-learn, excluding those that require
  special arguments (such as "base_estimator" or "estimators") or are known to fail with certain inputs.
  If a specific list of regressor names is provided, only those regressors are returned.

  Parameters:
    regList (list, optional): List of regressor names to include. If empty or None, all available regressors are considered.

  Returns:
    list: List of tuples containing regressor names and their corresponding class objects.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    
    # Get all filtered regressors.
    regressors = mlh.GetFilteredRegressors()
    for name, reg in regressors:
      print(f"Regressor Name: {name}, Class: {reg}")
    # Get only specific regressors.
    regressors = mlh.GetFilteredRegressors(["RandomForestRegressor", "LinearRegression"])
    for name, reg in regressors:
      print(f"Regressor Name: {name}, Class: {reg}")
  '''

  from sklearn.utils import all_estimators

  if ((regList is None) or (len(regList) <= 0)):
    estimators = all_estimators(type_filter="regressor")
    args = ["base_estimator", "estimators"]

    # Known problematic regressors or those requiring special arguments
    toIgnore = [
      est
      for est in estimators
      for varName in est[1].__init__.__code__.co_varnames
      if (varName in args)
    ]
    toIgnore = toIgnore + [
      "MultiOutputRegressor",
      "RegressorChain",
      "StackingRegressor",
      "VotingRegressor",
      "RadiusNeighborsRegressor",
      "IsotonicRegression",  # Needs 1d input
      "MultiTaskElasticNet",
      "MultiTaskElasticNetCV",
      "MultiTaskLasso",
      "MultiTaskLassoCV",
      "ClassifierChain",  # Not a regressor, but sometimes appears
    ]

    estimators = [
      est
      for est in estimators
      if (est[0] not in toIgnore)
    ]

    for i in range(len(estimators)):
      print(f"{i + 1}: {estimators[i][0]}", flush=True)

  else:
    estimators = []
    for reg in regList:
      estimators.append(
        [
          reg,
          [
            est for est in all_estimators(type_filter="regressor")
            if (est[0] == reg)
          ][0][1]
        ]
      )

  return estimators


def GetClassifierClassByName(clsName):
  r'''
  Retrieve a scikit-learn classifier class by its name.

  This function returns the classifier class object corresponding to the specified name from scikit-learn.
  If the classifier is not found, None is returned.

  Parameters:
    clsName (str): Name of the classifier to retrieve.

  Returns:
    class: Classifier class object corresponding to the specified name, or None if not found.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    # Get the RandomForestClassifier class.
    rfCls = mlh.GetClassifierClassByName("RandomForestClassifier")
    print(rfCls)
    # Get a non-existent classifier.
    noneCls = mlh.GetClassifierClassByName("NonExistentClassifier")
    print(noneCls)  # None.
  '''

  from sklearn.utils import all_estimators

  # Retrieve all available classifiers in scikit-learn.
  estimators = all_estimators(type_filter="classifier")

  # Filter the classifiers based on the provided name.
  for est in estimators:
    if (est[0] == clsName):
      return est[1]

  # Return None if the classifier is not found.
  return None


def GetRegressorClassByName(regName):
  r'''
  Retrieve a scikit-learn regressor class by its name.

  This function returns the regressor class object corresponding to the specified name from scikit-learn.
  If the regressor is not found, None is returned.

  Parameters:
    regName (str): Name of the regressor to retrieve.

  Returns:
    class: Regressor class object corresponding to the specified name, or None if not found.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    # Get the RandomForestRegressor class.
    rfReg = mlh.GetRegressorClassByName("RandomForestRegressor")
    print(rfReg)
    # Get a non-existent regressor.
    noneReg = mlh.GetRegressorClassByName("NonExistentRegressor")
    print(noneReg)  # None.
  '''

  from sklearn.utils import all_estimators

  # Retrieve all available regressors in scikit-learn.
  estimators = all_estimators(type_filter="regressor")

  # Filter the regressors based on the provided name.
  for est in estimators:
    if (est[0] == regName):
      return est[1]

  # Return None if the regressor is not found.
  return None


def GetMLClassificationModelObject(modelName, hyperparameters={}):
  r'''
  Get a machine learning classification model object based on the given name and hyperparameters.

  This function returns an instance of a classification model from scikit-learn or other libraries
  according to the provided model name and optional hyperparameters.

  Supported models include:
    - "MLP": Multi-layer Perceptron Classifier
    - "RF": Random Forest Classifier
    - "AB": AdaBoost Classifier
    - "KNN": K-Nearest Neighbors Classifier
    - "DT": Decision Tree Classifier
    - "SVC": Support Vector Classifier
    - "GNB": Gaussian Naive Bayes
    - "LR": Logistic Regression
    - "SGD": Stochastic Gradient Descent Classifier
    - "GB": Gradient Boosting Classifier
    - "Bagging": Bagging Classifier
    - "ETs": Extra Trees Classifier
    - "XGB": eXtreme Gradient Boosting Classifier
    - "LGBM": Light Gradient Boosting Machine Classifier
    - "Voting": Voting Classifier
    - "Stacking": Stacking Classifier
    - "CatBoost": CatBoost Classifier
    - "HGB": HistGradientBoosting Classifier

  Parameters:
    modelName (str): Name of the classification model to retrieve.
      Supported values include: "MLP", "RF", "AB", "KNN", "DT", "SVC", "GNB", "LR", "SGD", "GB", "Bagging",
      "ETs", "XGB", "LGBM", "Voting", "Stacking", "CatBoost", "HGB".
    hyperparameters (dict, optional): Dictionary of hyperparameters to pass to the model constructor.

  Returns:
    object: An instance of the requested classification model.

  Raises:
    ValueError: If an invalid model name is provided.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.MachineLearningHelper as mlh
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset.
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=0.2,
      random_state=np.random.randint(0, 10000),
    )
    # Get a scaler object (e.g., Standard Scaler).
    scaler = mlh.GetScalerObject("Standard")
    # Fit the scaler on the training data and transform both training and testing data.
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    # Get a classification model object (e.g., Random Forest).
    model = mlh.GetMLClassificationModelObject("RF", hyperparameters={"n_estimators": 100})
    # Train the model on the training data.
    model.fit(xTrain, yTrain)
    # Make predictions on the testing data.
    yPred = model.predict(xTest)
    # Calculate and print the accuracy of the model.
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")
  '''

  # Support common full class names in addition to short codes
  if (modelName in ["MLP", "MLPClassifier"]):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(**hyperparameters)
  elif (modelName in ["RF", "RandomForestClassifier"]):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**hyperparameters)
  elif (modelName in ["AB", "AdaBoostClassifier"]):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(**hyperparameters)
  elif (modelName in ["KNN", "KNeighborsClassifier"]):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(**hyperparameters)
  elif (modelName in ["DT", "DecisionTreeClassifier"]):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(**hyperparameters)
  elif (modelName in ["SVC", "SVC"]):
    from sklearn.svm import SVC
    return SVC(**hyperparameters)
  elif (modelName in ["GNB", "GaussianNB"]):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(**hyperparameters)
  elif (modelName in ["LR", "LogisticRegression"]):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**hyperparameters)
  elif (modelName in ["SGD", "SGDClassifier"]):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(**hyperparameters)
  elif (modelName in ["GB", "GradientBoostingClassifier"]):
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(**hyperparameters)
  elif (modelName in ["Bagging", "BaggingClassifier"]):
    from sklearn.ensemble import BaggingClassifier
    return BaggingClassifier(**hyperparameters)
  elif (modelName in ["ETs", "ExtraTreesClassifier"]):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(**hyperparameters)
  elif (modelName in ["XGB", "XGBClassifier"]):
    from xgboost import XGBClassifier
    return XGBClassifier(**hyperparameters)
  elif (modelName in ["LGBM", "LGBMClassifier"]):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**hyperparameters)
  elif (modelName == "Voting"):
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    estimators = [
      RandomForestClassifier(),
      DecisionTreeClassifier(),
    ]
    return VotingClassifier(estimators, **hyperparameters)
  elif (modelName == "Stacking"):
    from sklearn.ensemble import StackingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    estimators = [
      RandomForestClassifier(),
      DecisionTreeClassifier(),
    ]
    return StackingClassifier(estimators, **hyperparameters)
  elif (modelName == "CatBoost"):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**hyperparameters)
  elif (modelName == "HGB"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**hyperparameters)
  # You can add more models as needed.
  else:
    raise ValueError("Invalid model name.")


def GetClassifierModelSpace(classifierName, noOfFeatures=None, noOfSamples=None):
  r'''
  Get the hyperparameter search space for a given classifier.

  This function returns a dictionary representing the hyperparameter search space for a specified classifier.
  The search space is defined based on common hyperparameters for each classifier type, and can be adjusted
  based on the number of features and samples in the dataset.

  Parameters:
    classifierName (str): The name of the classifier for which to retrieve the hyperparameter search space.
      Supported values include "DT", "SVM", "LR", "RF", "KNN", "LGBM", "XGB", "HGB", "GB", "AdaBoost", "CatBoost",
      and "MLP".
    noOfFeatures (int, optional): The number of features in the dataset. This can be used to adjust the search
      space for certain hyperparameters (e.g., max_depth). If not provided, default values will be used.
    noOfSamples (int, optional): The number of samples in the dataset. This can be used to adjust the search space
     for certain hyperparameters (e.g., n_neighbors for KNN). If not provided, default values will be used.

  Returns:
    dict: A dictionary representing the hyperparameter search space for the specified classifier.
  '''

  # Assign a default value to noOfFeatures if it is not provided.
  if (noOfFeatures is None):
    noOfFeatures = 50
  # Assign a default value to noOfSamples if it is not provided.
  if (noOfSamples is None):
    noOfSamples = 1000

  # Ensure noOfFeatures is at least 1 to prevent empty ranges.
  noOfFeatures = max(1, noOfFeatures)
  # Ensure noOfSamples is at least 2 to prevent empty ranges.
  noOfSamples = max(2, noOfSamples)

  # Check if the classifier name corresponds to Decision Tree.
  if (classifierName == "DT"):
    # Define the search space for Decision Tree parameters.
    modelSpace = {
      "max_depth"        : list(range(1, noOfFeatures + 1)),
      "splitter"         : ["best", "random"],
      "criterion"        : ["gini", "entropy", "log_loss"],
      "min_samples_split": [2, 5, 10],
      "min_samples_leaf" : [1, 2, 4],
    }
  # Check if the classifier name corresponds to Support Vector Machine.
  elif (classifierName == "SVM"):
    # Define the search space for Support Vector Machine parameters.
    modelSpace = {
      "C"     : [0.01, 0.1, 1.0, 10.0, 100.0],
      "kernel": ["linear", "rbf", "poly", "sigmoid"],
      "gamma" : ["scale", "auto"],
      "degree": [2, 3, 4],
    }
  # Check if the classifier name corresponds to Logistic Regression.
  elif (classifierName == "LR"):
    # Define the search space for Logistic Regression parameters.
    modelSpace = {
      "C"       : [0.01, 0.1, 1.0, 10.0, 100.0],
      "penalty" : ["l1", "l2", "elasticnet"],
      "solver"  : ["liblinear", "lbfgs", "saga"],
      "max_iter": [100, 500, 1000],
    }
  # Check if the classifier name corresponds to Random Forest.
  elif (classifierName == "RF"):
    # Define the search space for Random Forest parameters.
    modelSpace = {
      "n_estimators"     : list(range(50, 501, 50)),
      "max_depth"        : list(range(1, noOfFeatures + 1)),
      "criterion"        : ["gini", "entropy", "log_loss"],
      "max_features"     : ["sqrt", "log2", None],
      "min_samples_split": [2, 5, 10],
    }
  # Check if the classifier name corresponds to K-Nearest Neighbors.
  elif (classifierName == "KNN"):
    # Calculate the maximum number of neighbors based on sample size.
    maxNeighbors = max(2, int(noOfSamples / 2.0))
    # Define the search space for K-Nearest Neighbors parameters.
    modelSpace = {
      "n_neighbors": list(range(1, maxNeighbors)),
      "weights"    : ["uniform", "distance"],
      "algorithm"  : ["ball_tree", "kd_tree", "brute"],
      "metric"     : ["minkowski", "euclidean", "manhattan"],
    }
  # Check if the classifier name corresponds to LightGBM.
  elif (classifierName == "LGBM"):
    # Define the search space for LightGBM parameters.
    modelSpace = {
      "n_estimators"    : list(range(50, 501, 50)),
      "num_leaves"      : list(range(10, 100, 5)),
      "max_depth"       : list(range(-1, noOfFeatures + 1)),
      "learning_rate"   : [0.01, 0.05, 0.1, 0.3],
      "subsample"       : [0.6, 0.8, 1.0],
      "colsample_bytree": [0.6, 0.8, 1.0],
    }
  # Check if the classifier name corresponds to XGBoost.
  elif (classifierName == "XGB"):
    # Define the search space for XGBoost parameters.
    modelSpace = {
      "n_estimators"    : list(range(50, 501, 50)),
      "max_depth"       : list(range(3, 11)),
      "learning_rate"   : [0.01, 0.05, 0.1, 0.3],
      "subsample"       : [0.6, 0.8, 1.0],
      "colsample_bytree": [0.6, 0.8, 1.0],
      "min_child_weight": [1, 3, 5],
    }
  # Check if the classifier name corresponds to HistGradientBoosting.
  elif (classifierName == "HGB"):
    # Define the search space for HistGradientBoosting parameters.
    modelSpace = {
      "max_iter"         : [100, 200, 300],
      "max_leaf_nodes"   : list(range(15, 100)),
      "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
      "l2_regularization": [0.01, 0.1, 1.0, 10.0],
    }
  # Check if the classifier name corresponds to Gradient Boosting.
  elif (classifierName == "GB"):
    # Define the search space for Gradient Boosting parameters.
    modelSpace = {
      "n_estimators"     : list(range(50, 501, 50)),
      "max_depth"        : list(range(1, 11)),
      "learning_rate"    : [0.01, 0.05, 0.1, 0.3],
      "subsample"        : [0.6, 0.8, 1.0],
      "min_samples_split": [2, 5, 10],
    }
  # Check if the classifier name corresponds to AdaBoost.
  elif (classifierName == "AdaBoost"):
    # Define the search space for AdaBoost parameters.
    modelSpace = {
      "n_estimators" : list(range(50, 501, 50)),
      "learning_rate": [0.01, 0.1, 0.5, 1.0],
    }
  # Check if the classifier name corresponds to CatBoost.
  elif (classifierName == "CatBoost"):
    # Define the search space for CatBoost parameters.
    modelSpace = {
      "iterations"   : list(range(50, 501, 50)),
      "depth"        : list(range(4, 11)),
      "learning_rate": [0.01, 0.05, 0.1, 0.3],
      "l2_leaf_reg"  : [1.0, 5.0, 10.0],
    }
  # Check if the classifier name corresponds to Multi-Layer Perceptron.
  elif (classifierName == "MLP"):
    # Generate a list of tuple options for hidden layer sizes.
    layerOptions = [(i,) for i in range(16, 257, 16)] + [(i, i // 2) for i in range(32, 257, 32)]
    # Define the search space for Multi-Layer Perceptron parameters.
    modelSpace = {
      "activation"        : ["logistic", "relu", "tanh"],
      "solver"            : ["lbfgs", "sgd", "adam"],
      "alpha"             : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
      "hidden_layer_sizes": layerOptions,
      "learning_rate_init": [0.0001, 0.001, 0.01, 0.1],
    }
  # Check if the classifier name corresponds to Extra Trees.
  elif (classifierName == "ET"):
    # Define the search space for Extra Trees parameters.
    modelSpace = {
      "n_estimators": list(range(50, 501, 50)),
      "max_depth"   : list(range(1, noOfFeatures + 1)),
      "criterion"   : ["gini", "entropy", "log_loss"],
      "max_features": ["sqrt", "log2", None],
    }
  # Handle the case where the classifier name is invalid.
  else:
    # Raise an error if the classifier name does not match any known option.
    raise ValueError("Invalid classifier name.")
  # Return the defined model search space dictionary.

  return modelSpace


def GetMLRegressorModelObject(modelName, hyperparameters={}):
  r'''
  Get a machine learning regressor model object based on the given name and hyperparameters.

  This function returns an instance of a regression model from scikit-learn or other libraries
  according to the provided model name and optional hyperparameters.

  Supported models include:
    - "MLP": Multi-layer Perceptron Regressor
    - "RF": Random Forest Regressor
    - "AB": AdaBoost Regressor
    - "KNN": K-Nearest Neighbors Regressor
    - "DT": Decision Tree Regressor
    - "SVR": Support Vector Regressor
    - "LR": Linear Regression
    - "SGD": Stochastic Gradient Descent Regressor
    - "GB": Gradient Boosting Regressor
    - "Bagging": Bagging Regressor
    - "ETs": Extra Trees Regressor
    - "XGB": eXtreme Gradient Boosting Regressor
    - "LGBM": Light Gradient Boosting Machine Regressor
    - "Voting": Voting Regressor
    - "Stacking": Stacking Regressor
    - "CatBoost": CatBoost Regressor
    - "HGB": HistGradientBoosting Regressor

  Parameters:
    modelName (str): Name of the regression model to retrieve.
    hyperparameters (dict, optional): Dictionary of hyperparameters to pass to the model constructor.

  Returns:
    object: An instance of the requested regression model.

  Raises:
    ValueError: If an invalid model name is provided.
  '''

  if (modelName == "MLP"):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(**hyperparameters)
  elif (modelName == "RF"):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(**hyperparameters)
  elif (modelName == "AB"):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(**hyperparameters)
  elif (modelName == "KNN"):
    from sklearn.neighbors import KNeighborsRegressor
    return KNeighborsRegressor(**hyperparameters)
  elif (modelName == "DT"):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(**hyperparameters)
  elif (modelName == "SVR"):
    from sklearn.svm import SVR
    return SVR(**hyperparameters)
  elif (modelName == "LR"):
    from sklearn.linear_model import LinearRegression
    return LinearRegression(**hyperparameters)
  elif (modelName == "SGD"):
    from sklearn.linear_model import SGDRegressor
    return SGDRegressor(**hyperparameters)
  elif (modelName == "GB"):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(**hyperparameters)
  elif (modelName == "Bagging"):
    from sklearn.ensemble import BaggingRegressor
    return BaggingRegressor(**hyperparameters)
  elif (modelName == "ETs"):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(**hyperparameters)
  elif (modelName == "XGB"):
    from xgboost import XGBRegressor
    return XGBRegressor(**hyperparameters)
  elif (modelName == "LGBM"):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**hyperparameters)
  elif (modelName == "Voting"):
    from sklearn.ensemble import VotingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    estimators = [
      ("rf", RandomForestRegressor()),
      ("dt", DecisionTreeRegressor()),
    ]
    return VotingRegressor(estimators, **hyperparameters)
  elif (modelName == "Stacking"):
    from sklearn.ensemble import StackingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    estimators = [
      ("rf", RandomForestRegressor()),
      ("dt", DecisionTreeRegressor()),
    ]
    return StackingRegressor(estimators, **hyperparameters)
  elif (modelName == "CatBoost"):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**hyperparameters)
  elif (modelName == "HGB"):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**hyperparameters)
  else:
    raise ValueError("Invalid regressor model name.")


def PerformFeatureSelection(tech, fsRatio, xTrain, yTrain, xTest, yTest, returnFeatures=False):
  r'''
  Perform feature selection on training and testing data using the specified technique and ratio.

  This function applies feature selection or dimensionality reduction to the input data using
  techniques such as PCA, LDA, Random Forest importance, RFE, Chi2, Mutual Information, or ANOVA.

  Supported techniques include:
    - "PCA": Principal Component Analysis
    - "RF": Random Forest Feature Importance
    - "RFE": Recursive Feature Elimination
    - "Chi2": Chi-Squared Feature Selection
    - "MI": Mutual Information
    - "ANOVA": ANOVA F-value
    - "LDA": Linear Discriminant Analysis

  Parameters:
    tech (str): Feature selection technique to use.
      Supported values include: "PCA", "RF", "RFE", "Chi2", "MI", "ANOVA", "LDA".
    fsRatio (float): Ratio of features to select (0 < fsRatio <= 100).
    xTrain (numpy.ndarray or pandas.DataFrame): Training data.
    yTrain (numpy.ndarray or pandas.Series): Training labels.
    xTest (numpy.ndarray or pandas.DataFrame): Testing data.
    yTest (numpy.ndarray or pandas.Series): Testing labels.
    returnFeatures (bool, optional): If True, returns selected features and selector object.

  Returns:
    tuple: Transformed training and testing data, selector object, and selected features (if requested).
      - (xTrainFS, xTestFS): Transformed training and testing data after feature selection.
      - selector: The feature selection or dimensionality reduction object used.
      - features: List of selected feature names (if returnFeatures is True).

  Raises:
    ValueError: If the number of features is invalid or the technique is not supported.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.MachineLearningHelper as mlh
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset.
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=0.2,
      random_state=np.random.randint(0, 10000),
    )
    # Perform feature selection using PCA to select 50% of features.
    xTrainFS, xTestFS, selector, features = mlh.PerformFeatureSelection(
      "PCA", 50, xTrain, yTrain, xTest, yTest, returnFeatures=True
    )
    # Print the selected features.
    print(f"Selected Features: {features}")
    # Print the shape of the transformed training and testing data.
    print(f"xTrainFS shape: {xTrainFS.shape}, xTestFS shape: {xTestFS.shape}")
    # Get a scaler object (e.g., Standard Scaler).
    scaler = mlh.GetScalerObject("Standard")
    # Fit the scaler on the balanced training data and transform both training and testing data.
    xTrainFS = scaler.fit_transform(xTrainFS)
    xTestFS = scaler.transform(xTestFS)
    # Initialize and train a Logistic Regression model.
    model = mlh.GetMLClassificationModelObject("LR")
    model.fit(xTrainFS, yTrain)
    # Make predictions on the testing data.
    yPred = model.predict(xTestFS)
    # Calculate and print the accuracy of the model.
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")
  '''

  # Calculate the number of features to select based on the ratio provided.
  noOfFeatures = int(fsRatio * xTrain.shape[1] / 100.0)
  # Bound the number of features to be at least 1 and at most the number of features in the dataset.
  noOfFeatures = max(1, min(noOfFeatures, xTrain.shape[1]))

  # For PCA, n_components must be <= min(n_samples, n_features)
  if (tech == "PCA"):
    maxComponents = min(xTrain.shape[0], xTrain.shape[1])
    noOfFeatures = min(noOfFeatures, maxComponents)

  # For LDA, n_components must be <= min(n_features, n_classes - 1, n_samples)
  if (tech == "LDA"):
    numClasses = len(np.unique(yTrain))
    maxComponents = min(xTrain.shape[1], numClasses - 1, xTrain.shape[0])
    noOfFeatures = min(noOfFeatures, maxComponents)

  # Raise an error if the number of features exceeds the number of features in the dataset.
  if (noOfFeatures > xTrain.shape[1]):
    raise ValueError("Number of features must be less than or equal to the number of features in the dataset.")

  # If the number of features equals the total number of features,
  # return the original data without feature selection.
  if (noOfFeatures == xTrain.shape[1]):
    if (returnFeatures):
      features = (
        xTrain.columns.tolist()
        if (hasattr(xTrain, "columns"))
        else [f"Feature_{i + 1}" for i in range(xTrain.shape[1])]
      )
      return xTrain, xTest, None, features
    return xTrain, xTest, None

  # Perform PCA for dimensionality reduction if the specified technique is "PCA".
  if (tech == "PCA"):
    from sklearn.decomposition import PCA  # Import PCA from sklearn.
    fs = PCA(n_components=noOfFeatures)  # Initialize PCA with the specified number of components.
    xTrainFS = fs.fit_transform(xTrain)  # Fit PCA on the training data and transform it.
    xTestFS = fs.transform(xTest)  # Transform the testing data using the fitted PCA.
    features = [
      f"PCA_{i + 1}" for i in range(noOfFeatures)
    ]

  # Perform feature selection using Random Forest feature importance if the specified technique is "RF".
  elif (tech == "RF"):
    from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier.
    fs = RandomForestClassifier()  # Initialize a Random Forest classifier.
    fs.fit(xTrain, yTrain)  # Fit the Random Forest model on the training data.
    importances = fs.feature_importances_  # Retrieve feature importances from the trained model.
    indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order.
    if (hasattr(xTrain, "columns")):
      trainCols = xTrain.columns[indices[:noOfFeatures]]  # Select the top features from the training data.
      testCols = xTest.columns[indices[:noOfFeatures]]  # Select the top features from the testing data.
      xTrainFS = xTrain[trainCols]  # Filter the training data to keep only the selected features.
      xTestFS = xTest[testCols]  # Filter the testing data to keep only the selected features.
      features = trainCols.tolist()  # Convert to list for consistency.
    else:
      xTrainFS = xTrain[:, indices[:noOfFeatures]]
      xTestFS = xTest[:, indices[:noOfFeatures]]
      features = [f"Feature_{i + 1}" for i in range(noOfFeatures)]

  # Perform Recursive Feature Elimination (RFE) if the specified technique is "RFE".
  elif (tech == "RFE"):
    from sklearn.feature_selection import RFE  # Import RFE from sklearn.
    from sklearn.ensemble import RandomForestClassifier  # Import Random Forest classifier.
    if (hasattr(xTrain, "columns")):
      columns = xTrain.columns  # Get the column names from the training data.
    else:
      columns = [f"Feature_{i + 1}" for i in range(xTrain.shape[1])]
    # Initialize RFE with a Random Forest estimator.
    fs = RFE(RandomForestClassifier(), n_features_to_select=noOfFeatures)
    fs.fit(xTrain, yTrain)  # Fit RFE on the training data.
    xTrainFS = fs.transform(xTrain)  # Transform the training data using the fitted RFE.
    xTestFS = fs.transform(xTest)  # Transform the testing data using the fitted RFE.
    features = [columns[i] for i, selected in enumerate(fs.support_) if selected]

  # Perform feature selection using Chi-squared test if the specified technique is "Chi2".
  elif (tech == "Chi2"):
    # Import SelectKBest and chi2 from sklearn.
    from sklearn.feature_selection import SelectKBest, chi2
    # Ensure input is non-negative for chi2
    if (hasattr(xTrain, "columns")):
      xTrainNonNeg = xTrain.clip(lower=0)
      xTestNonNeg = xTest.clip(lower=0)
      columns = xTrain.columns
    else:
      xTrainNonNeg = np.clip(xTrain, 0, None)
      xTestNonNeg = np.clip(xTest, 0, None)
      columns = [f"Feature_{i + 1}" for i in range(xTrain.shape[1])]
    fs = SelectKBest(chi2, k=noOfFeatures)  # Initialize SelectKBest with the Chi-squared test.
    xTrainFS = fs.fit_transform(xTrainNonNeg, yTrain)  # Fit SelectKBest on the training data and transform it.
    xTestFS = fs.transform(xTestNonNeg)  # Transform the testing data using the fitted SelectKBest.
    features = [columns[i] for i, selected in enumerate(fs.get_support()) if selected]

  # Perform feature selection using Mutual Information if the specified technique is "MI".
  elif (tech == "MI"):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    if (hasattr(xTrain, "columns")):
      columns = xTrain.columns
    else:
      columns = [f"Feature_{i + 1}" for i in range(xTrain.shape[1])]
    fs = SelectKBest(mutual_info_classif, k=noOfFeatures)  # Initialize SelectKBest with Mutual Information.
    xTrainFS = fs.fit_transform(xTrain, yTrain)  # Fit SelectKBest on the training data and transform it.
    xTestFS = fs.transform(xTest)  # Transform the testing data using the fitted SelectKBest.
    features = [columns[i] for i, selected in enumerate(fs.get_support()) if selected]

  # Perform feature selection using ANOVA if the specified technique is "ANOVA".
  elif (tech == "ANOVA"):
    # Import SelectKBest and f_classif from sklearn.
    from sklearn.feature_selection import SelectKBest, f_classif
    if (hasattr(xTrain, "columns")):
      columns = xTrain.columns
    else:
      columns = [f"Feature_{i + 1}" for i in range(xTrain.shape[1])]
    fs = SelectKBest(f_classif, k=noOfFeatures)  # Initialize SelectKBest with ANOVA F-value.
    xTrainFS = fs.fit_transform(xTrain, yTrain)  # Fit SelectKBest on the training data and transform it.
    xTestFS = fs.transform(xTest)  # Transform the testing data using the fitted SelectKBest.
    features = columns[fs.get_support()]  # Get the selected features from SelectKBest.

  # Perform feature selection using Linear Discriminant Analysis if the specified technique is "LDA".
  elif (tech == "LDA"):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import LDA from sklearn.
    # Initialize LDA with the specified number of components.
    # n_components cannot be larger than min(n_features, n_classes - 1)
    # Ensure noOfFeatures is within valid bounds.
    noOfFeatures = min(noOfFeatures, xTrain.shape[1], len(np.unique(yTrain)) - 1)
    fs = LinearDiscriminantAnalysis(n_components=noOfFeatures)
    xTrainFS = fs.fit_transform(xTrain, yTrain)  # Fit LDA on the training data and transform it.
    xTestFS = fs.transform(xTest)  # Transform the testing data using the fitted LDA.
    features = [f"LDA_{i + 1}" for i in range(noOfFeatures)]

  else:
    raise ValueError(f"Invalid feature selection technique ({tech}) specified.")

  # Return the transformed training and testing data after feature selection.
  if (returnFeatures):
    return xTrainFS, xTestFS, fs, features
  return xTrainFS, xTestFS, fs


def PerformFeatureRanking(trainDF, testDF, columns, rankTechStr, noOfFeatures):
  r'''
  Apply feature ranking to training and test DataFrames and return reduced DataFrames.

  This function ranks features using a specified ranking technique and selects the top
  percentage or absolute number of features specified by noOfFeatures.
  Supported ranking techniques include: "UDFS", "MCFS", "MRMR", "CMIM", "JMI", "MIM",
  "MIFS", "CIFE", "ICAP", "DISR".

  Parameters:
    trainDF (pandas.DataFrame): The training data as a DataFrame.
    testDF (pandas.DataFrame): The test data as a DataFrame.
    columns (list): List of feature column names to be used for ranking.
    rankTechStr (str): The ranking technique to be used.
    noOfFeatures (int or float): The number of features to select. If <= 100, interpreted as a percentage.

  Returns:
    tuple: A tuple containing the reduced training DataFrame, reduced test DataFrame, and the list of selected columns.
  '''

  # Validate that trainDF is a pandas DataFrame.
  if (not isinstance(trainDF, pd.DataFrame)):
    raise ValueError("`trainDF` must be a pandas DataFrame.")
  # Validate that testDF is a pandas DataFrame.
  if (not isinstance(testDF, pd.DataFrame)):
    raise ValueError("`testDF` must be a pandas DataFrame.")
  # Extract feature matrix X and target y from training DataFrame.
  xTrain, yTrain = trainDF[columns], trainDF["activity"]
  # Extract feature matrix X and target y from test DataFrame.
  xTest, yTest = testDF[columns], testDF["activity"]
  # Initialize selectedColumns with all training columns by default.
  selectedColumns = xTrain.columns
  # Check if a ranking technique was provided.
  if (rankTechStr is not None):
    # Handle UDFS ranking technique.
    if (rankTechStr == "UDFS"):
      # Import UDFS module.
      from skfeature.function.sparse_learning_based import UDFS
      # Compute ordering using UDFS.
      ordering = UDFS.udfs(xTrain.values)
    # Handle MCFS ranking technique.
    elif (rankTechStr == "MCFS"):
      # Import MCFS module.
      from skfeature.function.sparse_learning_based import MCFS
      # Compute ordering using MCFS.
      ordering = MCFS.mcfs(xTrain.values)
    # Handle MRMR ranking technique.
    elif (rankTechStr == "MRMR"):
      # Import MRMR module.
      from skfeature.function.information_theoretical_based import MRMR
      # Compute ordering using MRMR and ignore other returned values.
      ordering, _, _ = MRMR.mrmr(xTrain.values, yTrain.values)
    # Handle CMIM ranking technique.
    elif (rankTechStr == "CMIM"):
      # Import CMIM module.
      from skfeature.function.information_theoretical_based import CMIM
      # Compute ordering using CMIM and ignore other returned values.
      ordering, _, _ = CMIM.cmim(xTrain.values, yTrain.values)
    # Handle JMI ranking technique.
    elif (rankTechStr == "JMI"):
      # Import JMI module.
      from skfeature.function.information_theoretical_based import JMI
      # Compute ordering using JMI and ignore other returned values.
      ordering, _, _ = JMI.jmi(xTrain.values, yTrain.values)
    # Handle MIM ranking technique.
    elif (rankTechStr == "MIM"):
      # Import MIM module.

      from skfeature.function.information_theoretical_based import MIM
      # Compute ordering using MIM and ignore other returned values.
      ordering, _, _ = MIM.mim(xTrain.values, yTrain.values)
    # Handle MIFS ranking technique.
    elif (rankTechStr == "MIFS"):
      # Import MIFS module.
      from skfeature.function.information_theoretical_based import MIFS
      # Compute ordering using MIFS and ignore other returned values.
      ordering, _, _ = MIFS.mifs(xTrain.values, yTrain.values)
    # Handle CIFE ranking technique.
    elif (rankTechStr == "CIFE"):
      # Import CIFE module.
      from skfeature.function.information_theoretical_based import CIFE
      # Compute ordering using CIFE and ignore other returned values.
      ordering, _, _ = CIFE.cife(xTrain.values, yTrain.values)
    # Handle ICAP ranking technique.
    elif (rankTechStr == "ICAP"):
      # Import ICAP module.

      from skfeature.function.information_theoretical_based import ICAP
      # Compute ordering using ICAP and ignore other returned values.
      ordering, _, _ = ICAP.icap(xTrain.values, yTrain.values)
    # Handle DISR ranking technique.
    elif (rankTechStr == "DISR"):
      # Import DISR module.
      from skfeature.function.information_theoretical_based import DISR
      # Compute ordering using DISR and ignore other returned values.
      ordering, _, _ = DISR.disr(xTrain.values, yTrain.values)
    # Raise error for unsupported ranking technique.
    else:
      raise ValueError("Invalid ranking technique specified.")
    # Print which ranking technique was applied.
    print("Applied feature ranking technique:", rankTechStr)
    # Convert noOfFeatures from percent to absolute count if it looks like a percentage.
    if (noOfFeatures <= 100):
      noOfFeaturesCount = int(np.round(noOfFeatures / 100.0 * xTrain.shape[1]))
    # Otherwise treat noOfFeatures as absolute count.
    else:
      noOfFeaturesCount = int(noOfFeatures)
    # Ensure at least one feature is selected and not more than available.
    noOfFeaturesCount = max(1, min(noOfFeaturesCount, xTrain.shape[1]))
    # Derive ordered column names from ordering indices.
    orderedCols = xTrain.columns[ordering]
    # Select the top columns based on the computed ordering.
    selectedColumns = orderedCols[:noOfFeaturesCount]
    # Reduce the training feature set to the selected columns.
    xTrain = xTrain[selectedColumns]
    # Reduce the test feature set to the selected columns.
    xTest = xTest[selectedColumns]
  # If no ranking technique was provided, keep all columns.
  else:
    # Preserve original columns as selectedColumns.
    selectedColumns = xTrain.columns
    # Ensure training DataFrame uses the selected columns.
    xTrain = xTrain[selectedColumns]
    # Ensure test DataFrame uses the selected columns.
    xTest = xTest[selectedColumns]
  # Concatenate the reduced training features with the target column.
  trainDF = pd.concat([xTrain, yTrain], axis=1)
  # Concatenate the reduced test features with the target column.
  testDF = pd.concat([xTest, yTest], axis=1)
  # Return the reduced DataFrames and the list of selected columns.
  return trainDF, testDF, selectedColumns


def PerformDataBalancing(xTrain, yTrain, techniqueStr="SMOTE"):
  r'''
  Perform data balancing on training data using the specified oversampling or undersampling technique.

  This function applies data balancing techniques such as SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE,
  KSMOTE, RandomOverSampler, RandomUnderSampler, NearMiss, NearMiss-1, NearMiss-2, NearMiss-3,
  TomekLinks, or ClusterCentroids to the input training data.

  Supported techniques include:
    - "SMOTE": Synthetic Minority Over-sampling Technique
    - "ADASYN": Adaptive Synthetic Sampling Approach
    - "BSMOTE": Borderline SMOTE
    - "SVMSMOTE": Support Vector Machine SMOTE
    - "KSMOTE": KMeans SMOTE
    - "ROS": Random Over-Sampling
    - "RUS": Random Under-Sampling
    - "NearMiss": Near Miss Sampling. This is the same as "NearMiss-1".
    - "NearMiss-1": Near Miss version 1
    - "NearMiss-2": Near Miss version 2
    - "NearMiss-3": Near Miss version 3
    - "TomekLinks": Tomek Links
    - "CCx": Cluster Centroids

  Parameters:
    xTrain (array-like): Training data features.
    yTrain (array-like): Training data labels.
    techniqueStr (str, optional): Name of the balancing technique to use.
      Supported values include: "SMOTE", "ADASYN", "BSMOTE", "SVMSMOTE", "KSMOTE", "ROS",
      "RUS", "NearMiss", "NearMiss-1", "NearMiss-2", "NearMiss-3",
      "TL", "CCx". Default is "SMOTE".

  Returns:
    tuple: Resampled training data features, labels, and balancing object.
      - xTrain: Resampled training data features.
      - yTrain: Resampled training data labels.
      - obj: Data balancing object used for resampling.

  Raises:
    ValueError: If an invalid balancing technique is specified.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.MachineLearningHelper as mlh
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    # Create an imbalanced dataset.
    X, y = make_classification(
      n_classes=2, class_sep=2, weights=[0.9, 0.1],
      n_informative=3, n_redundant=1, flip_y=0,
      n_features=20, n_clusters_per_class=1,
      n_samples=1000, random_state=np.random.randint(0, 10000),
    )
    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=0.2,
      random_state=np.random.randint(0, 10000),
    )
    # Perform data balancing using SMOTE.
    xTrainBalanced, yTrainBalanced, obj = mlh.PerformDataBalancing(xTrain, yTrain, techniqueStr="SMOTE")
    # Print the class distribution before and after balancing.
    print(f"Original class distribution: {np.bincount(yTrain)}")
    print(f"Balanced class distribution: {np.bincount(yTrainBalanced)}")
    # Get a scaler object (e.g., Standard Scaler).
    scaler = mlh.GetScalerObject("Standard")
    # Fit the scaler on the balanced training data and transform both training and testing data.
    xTrainBalanced = scaler.fit_transform(xTrainBalanced)
    xTest = scaler.transform(xTest)
    # Initialize and train a Logistic Regression model.
    model = mlh.GetMLClassificationModelObject("LR")
    model.fit(xTrainBalanced, yTrainBalanced)
    # Make predictions on the testing data.
    yPred = model.predict(xTest)
    # Calculate and print the accuracy of the model.
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")
  '''

  if (techniqueStr is None):
    return xTrain, yTrain, None

  if (techniqueStr == "SMOTE"):
    # Import SMOTE from imbalanced-learn.
    from imblearn.over_sampling import SMOTE
    technique = SMOTE
  elif (techniqueStr == "ADASYN"):
    # Import ADASYN from imbalanced-learn.
    from imblearn.over_sampling import ADASYN
    technique = ADASYN
  elif (techniqueStr == "BSMOTE"):
    # Import BorderlineSMOTE from imbalanced-learn.
    from imblearn.over_sampling import BorderlineSMOTE
    technique = BorderlineSMOTE
  elif (techniqueStr == "KSMOTE"):
    # Import KMeansSMOTE from imbalanced-learn.
    from imblearn.over_sampling import KMeansSMOTE
    technique = KMeansSMOTE
  elif (techniqueStr == "SVMSMOTE"):
    # Import SVMSMOTE from imbalanced-learn.
    from imblearn.over_sampling import SVMSMOTE
    technique = SVMSMOTE
  elif (techniqueStr == "ROS"):
    # Import RandomOverSampler from imbalanced-learn.
    from imblearn.over_sampling import RandomOverSampler
    technique = RandomOverSampler
  elif (techniqueStr == "RUS"):
    # Import RandomUnderSampler from imbalanced-learn.
    from imblearn.under_sampling import RandomUnderSampler
    technique = RandomUnderSampler
  elif (techniqueStr == "NearMiss"):
    # Import NearMiss from imbalanced-learn.
    from imblearn.under_sampling import NearMiss
    technique = NearMiss
  elif (techniqueStr == "NearMiss-1"):
    # Import NearMiss from imbalanced-learn.
    from imblearn.under_sampling import NearMiss
    return lambda: NearMiss(version=1)
  elif (techniqueStr == "NearMiss-2"):
    # Import NearMiss from imbalanced-learn.
    from imblearn.under_sampling import NearMiss
    return lambda: NearMiss(version=2)
  elif (techniqueStr == "NearMiss-3"):
    # Import NearMiss from imbalanced-learn.
    from imblearn.under_sampling import NearMiss
    return lambda: NearMiss(version=3)
  elif (techniqueStr == "TL"):
    # Import TomekLinks from imbalanced-learn.
    from imblearn.under_sampling import TomekLinks
    technique = TomekLinks
  elif (techniqueStr == "CCx"):
    # Import ClusterCentroids from imbalanced-learn.
    from imblearn.under_sampling import ClusterCentroids
    technique = ClusterCentroids
  else:
    raise ValueError(f"Invalid data sampling technique ({techniqueStr}) specified.")

  params = {}
  if (techniqueStr in ["SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE"]):
    params = {
      "sampling_strategy": "minority",
      "random_state"     : np.random.randint(0, 10000),
    }

  # Create an instance of the specified sampling technique with the given parameters.
  obj = technique(**params)

  # Fit the sampling technique on the training data and resample it.
  xTrain, yTrain = obj.fit_resample(xTrain, yTrain)

  # Convert the resampled data back to a DataFrame if it was originally a DataFrame.
  return xTrain, yTrain, obj


def PerformOutlierDetection(
    xTrain,
    techniqueStr="IQR",
    contamination=0.05,
    returnMask=False,
):
  r'''
  Detect and optionally remove outliers from training data using the specified technique.

  This function applies outlier detection techniques such as IQR (Interquartile Range), Z-Score,
  Isolation Forest, Local Outlier Factor, Elliptic Envelope, One-Class SVM, DBSCAN, or Mahalanobis Distance
  to the input training data. It returns the filtered data and optionally a mask indicating which samples are considered inliers.

  Supported techniques include:
    - "IQR": Interquartile Range method
    - "ZScore": Z-Score method
    - "IForest": Isolation Forest
    - "LOF": Local Outlier Factor
    - "EllipticEnvelope": Elliptic Envelope (Gaussian)
    - "OCSVM": One-Class SVM
    - "DBSCAN": DBSCAN clustering (outliers as noise)
    - "Mahalanobis": Mahalanobis Distance

  Parameters:
    xTrain (array-like): Training data features (numpy array or pandas DataFrame).
    techniqueStr (str, optional): Name of the outlier detection technique to use.
      Supported values: "IQR", "ZScore", "IForest", "LOF", "EllipticEnvelope", "OCSVM", "DBSCAN", "Mahalanobis". Default is "IQR".
    contamination (float, optional): Proportion of outliers in the data (used for IForest/LOF/EllipticEnvelope/OCSVM/DBSCAN). Default is 0.05.
    returnMask (bool, optional): If True, also return a boolean mask of inliers. Default is False.

  Returns:
    xFiltered: Training data with outliers removed.
    mask (optional): Boolean mask indicating inliers (if returnMask=True).

  Raises:
    ValueError: If an invalid technique is specified.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh
    import numpy as np

    X = np.random.randn(100, 5)
    xFiltered = mlh.PerformOutlierDetection(X, techniqueStr="ZScore")
  '''

  if (techniqueStr is None):
    if (returnMask):
      return xTrain, np.ones(xTrain.shape[0], dtype=bool)
    return xTrain

  if (isinstance(xTrain, pd.DataFrame)):
    X = xTrain.values
  else:
    X = np.array(xTrain)

  if (techniqueStr == "IQR"):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = np.all((X >= lower) & (X <= upper), axis=1)

  elif (techniqueStr == "ZScore"):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    zScores = np.abs((X - mean) / std)
    mask = np.all(zScores < 3, axis=1)

  elif (techniqueStr == "IForest"):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(
      contamination=contamination,
      random_state=np.random.randint(0, 10000)
    )
    mask = clf.fit_predict(X) == 1

  elif (techniqueStr == "LOF"):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(contamination=contamination)
    mask = clf.fit_predict(X) == 1

  elif (techniqueStr == "EllipticEnvelope"):
    from sklearn.covariance import EllipticEnvelope
    clf = EllipticEnvelope(
      contamination=contamination,
      random_state=np.random.randint(0, 10000)
    )
    mask = clf.fit_predict(X) == 1

  elif (techniqueStr == "OCSVM"):
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(nu=contamination, kernel="rbf")
    mask = clf.fit_predict(X) == 1

  elif (techniqueStr == "DBSCAN"):
    from sklearn.cluster import DBSCAN
    # eps and min_samples can be tuned, here we use contamination to estimate min_samples.
    minSamples = max(2, int(contamination * X.shape[0]))
    clf = DBSCAN(eps=0.5, min_samples=minSamples)
    labels = clf.fit_predict(X)
    # DBSCAN labels -1 as outliers.
    mask = labels != -1

  elif (techniqueStr == "Mahalanobis"):
    from scipy.spatial import distance
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    covInv = np.linalg.pinv(cov)
    mahal = np.array([distance.mahalanobis(x, mean, covInv) for x in X])
    # Use 99th percentile as threshold for outliers.
    threshold = np.percentile(mahal, 100 * (1 - contamination))
    mask = mahal < threshold
  else:
    raise ValueError(f"Invalid outlier detection technique ({techniqueStr}) specified.")

  xFiltered = X[mask]
  if (isinstance(xTrain, pd.DataFrame)):
    xFiltered = xTrain.iloc[mask]
  if (returnMask):
    return xFiltered, mask
  return xFiltered


def MachineLearningClassification(
    datasetFilePath,  # Dataset file name (CSV format).
    scalerName,  # Name of the scaler to use.
    modelName,  # Name of the machine learning classification model.
    fsTechName,  # Feature selection technique name.
    fsTechRatio=0.2,  # Ratio of features to select.
    dataBalanceTech=None,  # Data balancing technique to be applied.
    outlierTech=None,  # Outlier detection technique to be applied.
    contamination=0.05,  # Proportion of outliers in the data (for outlier detection techniques).
    testRatio=0.2,  # Ratio of the test data.
    testFilePath=None,  # Optional test file for evaluation.
    targetColumn="Class",  # Name of the target column in the dataset.
    dropFirstColumn=True,  # Whether to drop the first column (usually an index or ID).
    dropNAColumns=True,  # Whether to drop columns with any null values.
    encodeCategorical=True,  # Whether to encode categorical features (if any).
    eps=1e-8,  # Small value to avoid division by zero (if needed).
):
  r'''
  Perform machine learning classification on a dataset with optional scaling,
  feature selection, and data balancing.

  This function loads a dataset from a CSV file, applies preprocessing steps
  (scaling, feature selection, data balancing), trains a classification model,
  and evaluates its performance. Optionally, a separate test file can be provided for evaluation.

  Data flow:
    Load Data -> Drop First Column (if specified) -> Drop NA Columns ->
    Encode Target Labels -> Encoder Categorial Features (if any and if needed) ->
    Split Features and Target -> Split Train/Test (if no test file) -> Outlier Detection (if specified) ->
    Scale Features -> Feature Selection -> Data Balancing (if specified) -> Train Model -> Evaluate Performance
    -> Plot Performance Metrics

  Parameters:
    datasetFilePath (str): Path to the dataset CSV file.
    scalerName (str): Name of the scaler to use (e.g., "Standard", "MinMax", "Robust").
    modelName (str): Name of the machine learning classification model (e.g., "LR" for Logistic Regression).
    fsTechName (str): Feature selection technique name.
    fsTechRatio (float): Ratio of features to select (default: 0.2).
    dataBalanceTech (str, optional): Data balancing technique to apply (e.g., "SMOTE").
    outlierTech (str, optional): Outlier detection technique to apply (e.g., "IQR").
    contamination (float, optional): Proportion of outliers in the data (used for outlier detection techniques).
    testRatio (float): Ratio of the test data (default: 0.2).
    testFilePath (str, optional): Optional path to a test CSV file for evaluation.
    targetColumn (str): Name of the target column in the dataset (default: "Class").
    dropFirstColumn (bool): Whether to drop the first column (usually an index or ID, default: True).
    dropNAColumns (bool): Whether to drop columns with any null values (default: True).
    encodeCategorical (bool): Whether to encode categorical features (if any, default: True).
    eps (float): Small value to avoid division by zero (if needed, default: 1e-8).

  Returns:
    tuple: A tuple containing:
      - dict: Dictionary of performance metrics.
      -  matplotlib.figure.Figure: Matplotlib figure object with performance plots.
      - dict: Dictionary containing various objects used in the process (e.g., model, scaler, feature selector). Keys include:
          - "Model": Trained machine learning model.
          - "CurrentColumns": List of current feature columns.
          - "Scaler": Scaler object used for scaling.
          - "FeatureSelector": Feature selection object used.
          - "FeatureSelectionTechnique": Name of the feature selection technique used.
          - "FeatureSelectionRatio": Ratio of features selected.
          - "SelectedFeatures": List of selected feature names.
          - "DataBalancingTechnique": Name of the data balancing technique used.
          - "DataBalancingObject": Data balancing object used.
          - "OutlierDetectionTechnique": Name of the outlier detection technique used.
          - "OutlierDetectionContamination": Contamination ratio used for outlier detection.
          - "OutlierDetectionMask": Outlier detection object used.
          - "LabelEncoder": Label encoder object used for encoding target labels.

  Examples
  --------
  .. code-block:: python

    import numpy as np
    import HMB.MachineLearningHelper as mlh
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset.
    data = load_iris()
    X = data.data
    y = data.target

    # Split the dataset into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=0.2,
      random_state=np.random.randint(0, 10000),
    )
    # Get a scaler object (e.g., Standard Scaler).
    scaler = mlh.GetScalerObject("Standard")
    # Fit the scaler on the training data and transform both training and testing data.
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.transform(xTest)
    # Get a classification model object (e.g., Random Forest).
    model = mlh.GetMLClassificationModelObject("RF", hyperparameters={"n_estimators": 100})
    # Train the model on the training data.
    model.fit(xTrain, yTrain)
    # Make predictions on the testing data.
    yPred = model.predict(xTest)
    # Calculate and print the accuracy of the model.
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.2f}")
  '''

  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import confusion_matrix
  import HMB.PerformanceMetrics as pm

  # Read the CSV file into a pandas DataFrame.
  data = pd.read_csv(datasetFilePath)

  # Check if dropFirstColumn is True, then drop the first column.
  if (dropFirstColumn):
    # Drop the first column if it is not the target column.
    if (data.columns[0] != targetColumn):
      data = data.drop(data.columns[0], axis=1)

  # Drop columns with any null values if dropNAColumns is True.
  if (dropNAColumns):
    # Drop empty columns from the DataFrame.
    # Updated from "all" to "any" to drop columns with any null values.
    # axis=1: means columns, how="any" means drop if any value is null.
    data = data.dropna(axis=1, how="any")

    # Drop rows with null or empty values from the DataFrame.
    # axis=0: means rows, how="any" means drop if any value is null.
    data = data.dropna(axis=0, how="any")

  # Features (X) are all columns except the "Class" column.
  X = data.drop(targetColumn, axis=1)
  # Store the current columns for feature selection.
  currentColumns = X.columns

  # Target (y) is the "Class" column.
  y = data[targetColumn]

  # Encode the target labels into numerical values using LabelEncoder.
  le = LabelEncoder()
  yEnc = le.fit_transform(y)
  labels = le.classes_

  # Encode categorical features if any and if encodeCategorical is True.
  featuresEncoders = {}
  # le = None
  if (encodeCategorical):
    categoricalCols = X.select_dtypes(include=["object", "category"]).columns
    if (len(categoricalCols) > 0):
      # Encode each categorical column using LabelEncoder.
      for col in categoricalCols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        featuresEncoders[col] = le

  if (testFilePath and os.path.exists(testFilePath)):
    # If a test file is provided, read it into a DataFrame.
    testData = pd.read_csv(testFilePath)

    # Check if dropFirstColumn is True, then drop the first column.
    if (dropFirstColumn):
      # Drop the first column if it is not the target column.
      if (testData.columns[0] != targetColumn):
        testData = testData.drop(testData.columns[0], axis=1)

    # Ensure test data has the same columns as training data.
    xTest = testData[currentColumns]

    # Target (y) is the "Class" column.
    y = testData[targetColumn]

    # Encode the target labels for the test data.
    yTest = le.transform(y)

    xTrain, yTrain = X, yEnc  # Use the original training data.
  else:
    # Split the data into training and testing sets.
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, yEnc,
      test_size=testRatio,
      random_state=np.random.randint(0, 10000),
      stratify=yEnc,
    )

  # Ensure that xTrain and xTest are DataFrames.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=currentColumns)

  # Perform outlier detection and removal on the training data.
  if (outlierTech is not None):  # Check if outlier detection technique is provided.
    xTrain, mask = PerformOutlierDetection(
      xTrain,  # Training data.
      techniqueStr=outlierTech,  # Outlier detection technique to use.
      contamination=contamination,  # Proportion of outliers in the data.
      returnMask=True,  # Return the mask of inliers.
    )
    yTrain = yTrain[mask]  # Filter the training labels using the mask.

  # Ensure that xTrain and xTest are DataFrames after outlier detection.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=currentColumns)

  if (scalerName is not None):
    # Create a scaler object to scale the features.
    scaler = GetScalerObject(scalerName)

    # Fit the scaler on the training data and transform it.
    xTrain = scaler.fit_transform(xTrain)

    # Transform the test data using the fitted scaler.
    xTest = scaler.transform(xTest)

    # Convert the scaled arrays back to DataFrames with the original column names.
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
    xTest = pd.DataFrame(xTest, columns=currentColumns)
  else:
    scaler = None

  # Ensure that xTrain and xTest are DataFrames after scaling.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=currentColumns)

  # Perform feature selection based on the specified technique.
  if (fsTechName is not None):
    xTrain, xTest, fs, selectedFeatures = PerformFeatureSelection(
      fsTechName,  # Feature selection technique.
      fsTechRatio,  # Ratio of features to select.
      xTrain,  # Training data.
      yTrain,  # Training labels.
      xTest,  # Testing data.
      yTest,  # Testing labels.
      returnFeatures=True,  # Return the features after feature selection.
    )
  else:
    fs = None
    # No feature selection, use all features.
    selectedFeatures = currentColumns

  # Ensure that xTrain and xTest are DataFrames after feature selection.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=selectedFeatures)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=selectedFeatures)

  # Perform data balancing of the training data.
  if (dataBalanceTech is not None):  # Check if data balancing technique is provided.
    xTrain, yTrain, dbObj = PerformDataBalancing(
      xTrain,  # Training data.
      yTrain,  # Training labels.
      techniqueStr=dataBalanceTech,  # Data balancing technique to use.
    )
  else:
    dbObj = None  # No data balancing technique used.

  # Ensure that xTrain is a DataFrame after data balancing.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=selectedFeatures)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=selectedFeatures)

  # Train a model on the training data.
  model = GetMLClassificationModelObject(modelName)
  model.fit(xTrain, yTrain)

  # Evaluate the model by making predictions on the test data.
  predTest = model.predict(xTest)

  # Calculate the confusion matrix using the true and predicted labels.
  cm = confusion_matrix(yTest, predTest)

  # Calculate performance metrics.
  metrics = pm.CalculatePerformanceMetrics(
    cm,  # Pass the confusion matrix.
    eps=eps,  # Small value to avoid division by zero.
    addWeightedAverage=True,  # Whether to include weighted averages in the output.
  )

  pltObject = pm.PlotConfusionMatrix(
    cm,  # Confusion matrix (2D list or numpy array).
    le.classes_,  # List of class labels.
    normalize=False,  # Whether to normalize the confusion matrix.
    roundDigits=3,  # Number of decimal places to round normalized values.
    title="Confusion Matrix",  # Title of the plot.
    cmap="Blues",  # Colormap for the plot.
    display=False,  # Whether to display the plot.
    save=False,  # Whether to save the plot.
    fileName=None,  # File name to save the plot.
    fontSize=15,  # Font size for labels and annotations.
    annotate=True,  # Whether to annotate cells with values.
    figSize=(8, 8),  # Figure size in inches.
    colorbar=True,  # Whether to show colorbar.
    returnFig=True,  # Whether to return the figure object.
  )

  # Create a dictionary to hold the objects for saving.
  objects = {
    "Model"                    : model,
    "CurrentColumns"           : currentColumns,
    "Scaler"                   : scaler,
    "ScalerName"               : scalerName,
    "FeatureSelector"          : fs,
    "FeatureSelectionTechnique": fsTechName,
    "FeatureSelectionRatio"    : fsTechRatio,
    "SelectedFeatures"         : selectedFeatures,
    "DataBalancingTechnique"   : dataBalanceTech,
    "DataBalancingObject"      : dbObj,
    "OutlierDetectionTechnique": outlierTech,
    "Contamination"            : contamination,
    "OutlierDetectionMask"     : dbObj,
    "LabelEncoder"             : le,
    "FeaturesEncoders"         : featuresEncoders,
    "Configurations"           : {
      "datasetFilePath"  : datasetFilePath,
      "scalerName"       : scalerName,
      "modelName"        : modelName,
      "fsTechName"       : fsTechName,
      "fsTechRatio"      : fsTechRatio,
      "dataBalanceTech"  : dataBalanceTech,
      "outlierTech"      : outlierTech,
      "contamination"    : contamination,
      "testRatio"        : testRatio,
      "testFilePath"     : testFilePath,
      "targetColumn"     : targetColumn,
      "dropFirstColumn"  : dropFirstColumn,
      "dropNAColumns"    : dropNAColumns,
      "encodeCategorical": encodeCategorical,
      "eps"              : eps,
    }
  }

  # Return the performance metrics, plot object, and objects for saving.
  return metrics, pltObject, objects


def MachineLearningRegression(
    datasetFilePath,  # Dataset file name (CSV format).
    scalerName,  # Name of the scaler to use.
    modelName,  # Name of the machine learning regression model.
    fsTechName,  # Feature selection technique name.
    fsTechRatio=0.2,  # Ratio of features to select.
    testRatio=0.2,  # Ratio of the test data.
    testFilePath=None,  # Optional test file for evaluation.
    targetColumn="Target",  # Name of the target column in the dataset.
    dropFirstColumn=True,  # Whether to drop the first column (usually an index or ID).
    dropNAColumns=True,  # Whether to drop columns with any null values.
    encodeCategorical=True,  # Whether to encode categorical features (if any).
):
  r'''
  Perform machine learning regression on a dataset with optional scaling and feature selection.

  This function loads a dataset from a CSV file, applies preprocessing steps
  (scaling, feature selection), trains a regression model, and evaluates its performance.
  Optionally, a separate test file can be provided for evaluation.

  Parameters:
    datasetFilePath (str): Path to the CSV dataset.
    scalerName (str): Name of the scaler to use.
    modelName (str): Name of the regression model.
    fsTechName (str): Feature selection technique.
    fsTechRatio (float): Ratio of features to select.
    testRatio (float): Test set ratio.
    testFilePath (str, optional): Path to test CSV file.
    targetColumn (str): Name of the target column.
    dropFirstColumn (bool): Drop first column if True.
    dropNAColumns (bool): Drop columns with NA if True.
    encodeCategorical (bool): Encode categorical features if True.

  Returns:
    tuple: A tuple containing:
      - dict: Dictionary of regression performance metrics (MSE, MAE, R2).
      - matplotlib.figure.Figure: Matplotlib figure object with regression results plot.
      - dict: Dictionary containing various objects used in the process (e.g., model, scaler, feature selector). Keys include:
          - "model": Trained regression model.
          - "scaler": Scaler object used for scaling (if any).
          - "feature_selector": Feature selection object used (if any).
          - "features": List of feature names used.
          - "metrics": Dictionary of performance metrics.

  Examples
  --------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh

    # Define the dataset file path and parameters.
    datasetFilePath = "path/to/your/dataset.csv"
    scalerName = "Standard"
    modelName = "LR"
    fsTechName = "PCA"
    fsTechRatio = 0.5  # Select 50% of features.
    testRatio = 0.2
    targetColumn = "Target"

    # Perform machine learning regression.
    metrics, pltObject, objects = mlh.MachineLearningRegression(
      datasetFilePath,
      scalerName,
      modelName,
      fsTechName,
      fsTechRatio,
      testRatio,
      targetColumn,
      dropFirstColumn=True,
    )
  '''

  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import HMB.PerformanceMetrics as pm

  # Read the CSV file into a pandas DataFrame.
  data = pd.read_csv(datasetFilePath)

  # Drop first column if specified.
  if (dropFirstColumn):
    data = data.iloc[:, 1:]

  # Drop columns with any null values if specified.
  if (dropNAColumns):
    data = data.dropna(axis=1)

  # Features (X) are all columns except the target column.
  X = data.drop(targetColumn, axis=1)
  currentColumns = X.columns

  # Target (y) is the target column.
  y = data[targetColumn]

  # Encode categorical features if any and if encodeCategorical is True.
  featuresEncoders = {}
  le = None
  if (encodeCategorical):
    for col in X.select_dtypes(include=["object", "category"]).columns:
      le = LabelEncoder()
      X[col] = le.fit_transform(X[col].astype(str))
      featuresEncoders[col] = le

  # Split train/test.
  if (testFilePath and os.path.exists(testFilePath)):
    testData = pd.read_csv(testFilePath)

    if (dropFirstColumn):
      testData = testData.iloc[:, 1:]

    if (dropNAColumns):
      testData = testData.dropna(axis=1)

    xTest = testData.drop(targetColumn, axis=1)
    yTest = testData[targetColumn]

    if (encodeCategorical):
      for col in xTest.select_dtypes(include=["object", "category"]).columns:
        if (col in featuresEncoders):
          xTest[col] = featuresEncoders[col].transform(xTest[col].astype(str))
        else:
          le = LabelEncoder()
          xTest[col] = le.fit_transform(xTest[col].astype(str))
    xTrain, yTrain = X, y
  else:
    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y,
      test_size=testRatio,
      random_state=np.random.randint(0, 10000),
    )

  # Ensure DataFrame.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=currentColumns)

  # Scaling.
  if (scalerName is not None):
    # Create a scaler object to scale the features.
    scaler = GetScalerObject(scalerName)

    # Fit the scaler on the training data and transform it.
    xTrain = scaler.fit_transform(xTrain)

    # Transform the test data using the fitted scaler.
    xTest = scaler.transform(xTest)

    # Convert the scaled arrays back to DataFrames with the original column names.
    xTrain = pd.DataFrame(xTrain, columns=currentColumns)
    xTest = pd.DataFrame(xTest, columns=currentColumns)
  else:
    scaler = None

  # Perform feature selection based on the specified technique.
  if (fsTechName is not None):
    xTrain, xTest, fs, selectedFeatures = PerformFeatureSelection(
      fsTechName,  # Feature selection technique.
      fsTechRatio,  # Ratio of features to select.
      xTrain,  # Training data.
      yTrain,  # Training labels.
      xTest,  # Testing data.
      yTest,  # Testing labels.
      returnFeatures=True,  # Return the features after feature selection.
    )
  else:
    fs = None
    # No feature selection, use all features.
    selectedFeatures = currentColumns

  # Ensure that xTrain and xTest are DataFrames after feature selection.
  if (not isinstance(xTrain, pd.DataFrame)):
    xTrain = pd.DataFrame(xTrain, columns=selectedFeatures)
  if (not isinstance(xTest, pd.DataFrame)):
    xTest = pd.DataFrame(xTest, columns=selectedFeatures)

  # Train regression model.
  model = GetMLRegressorModelObject(modelName)
  model.fit(xTrain, yTrain)

  # Predict and evaluate.
  predTest = model.predict(xTest)
  metrics = {
    "MSE": mean_squared_error(yTest, predTest),
    "MAE": mean_absolute_error(yTest, predTest),
    "R2" : r2_score(yTest, predTest),
  }

  # Optionally, plot predictions vs. true values.
  pltObject = pm.PlotRegressionResults(yTest, predTest)

  # Objects for saving.
  objects = {
    "Model"                    : model,
    "CurrentColumns"           : currentColumns,
    "Scaler"                   : scaler,
    "ScalerName"               : scalerName,
    "FeatureSelector"          : fs,
    "FeatureSelectionTechnique": fsTechName,
    "FeatureSelectionRatio"    : fsTechRatio,
    "SelectedFeatures"         : selectedFeatures,
    "OutlierDetectionMask"     : dbObj,
    "LabelEncoder"             : le,
    "FeaturesEncoders"         : featuresEncoders,
    "Configurations"           : {
      "datasetFilePath"  : datasetFilePath,
      "scalerName"       : scalerName,
      "modelName"        : modelName,
      "fsTechName"       : fsTechName,
      "fsTechRatio"      : fsTechRatio,
      "testRatio"        : testRatio,
      "testFilePath"     : testFilePath,
      "targetColumn"     : targetColumn,
      "dropFirstColumn"  : dropFirstColumn,
      "dropNAColumns"    : dropNAColumns,
      "encodeCategorical": encodeCategorical,
    }
  }

  return metrics, pltObject, objects


class OptunaTuningClassification(object):
  r'''
  Hyperparameter tuning for machine learning classification using Optuna.

  This class automates the process of hyperparameter optimization for machine learning pipelines
  using Optuna. It supports tuning over scalers, models, feature selection techniques and ratios,
  data balancing techniques, and outlier detection techniques. The class manages study creation,
  objective evaluation, result storage, and retrieval of the best parameters and study objects.

  Parameters:
    baseDir (str): Base directory where the dataset is stored.
    scalers (list): List of scalers to be used in the tuning process.
    models (list): List of machine learning models to be used in the tuning process.
    fsTechs (list): List of feature selection techniques to be used in the tuning process.
    fsRatios (list): List of feature selection ratios to be used in the tuning process.
    dataBalanceTechniques (list): List of data balancing techniques to be used in the tuning process.
    outliersTechniques (list): List of outlier detection techniques to be used in the tuning process.
    datasetFilename (str): Name of the dataset file to be used for training.
    storageFolderPath (str): Path to the folder where results will be stored.
    testFilename (str): Name of the test dataset file to be used for evaluation.
    testRatio (float, optional): Ratio of the test data to be used in the classification. Default is 0.2.
    contamination (float, optional): Proportion of outliers in the data (for outlier detection techniques). Default is 0.05.
    numTrials (int, optional): Number of trials for hyperparameter tuning. Default is 100.
    prefix (str, optional): Prefix for the study name and storage files. Default is "Optuna".
    samplerTech (str, optional): Sampler technique to be used for hyperparameter tuning. Options are "TPE", "Random", "NSGAIISampler", or "CmaEs". Default is "TPE".
    targetColumn (str, optional): Name of the target column in the dataset. Default is "Class".
    dropFirstColumn (bool, optional): Whether to drop the first column (usually an index or ID). Default is True.
    dropNAColumns (bool, optional): Whether to drop columns with any null values. Default is True.
    encodeCategorical (bool, optional): Whether to encode categorical features (if any). Default is True.
    saveFigures (bool, optional): Whether to save the performance plots. Default is True.
    eps (float, optional): Small value to avoid division by zero (if needed). Default is 1e-8.
    loadStudy (bool, optional): Whether to load an existing study if available. Default is True.
    verbose (bool, optional): Whether to print verbose output. Default is False.

  Attributes:
    study (optuna.study.Study): The Optuna study object.
    bestParams (dict): The best hyperparameters found by the study.
    bestValue (float): The best value found by the study.
    history (list): List of trial results and metrics.

  Example
  -------
  .. code-block:: python

    import HMB.MachineLearningHelper as mlh

    tuner = mlh.OptunaTuningClassification(
      baseDir="./data",
      scalers=["Standard", "MinMax"],
      models=["RF", "LR"],
      fsTechs=["PCA", "RF"],
      fsRatios=[50, 100],
      dataBalanceTechniques=["SMOTE", None],
      outliersTechniques=["IQR", None],
      datasetFilename="train.csv",
      storageFolderPath="./results",
      testFilename="test.csv",
      testRatio=0.2,
      contamination=0.05,
      numTrials=50,
      prefix="Optuna",
      samplerTech="TPE",
      targetColumn="Class",
      dropFirstColumn=True,
      dropNAColumns=True,
      encodeCategorical=True,
      saveFigures=True,
      eps=1e-8,
      loadStudy=False,
      verbose=True,
    )
    tuner.Tune()
    print(tuner.GetBestParams())
    print(tuner.GetBestValue())
  '''

  def __init__(
      self,
      baseDir,  # Base directory where the dataset is stored.
      scalers,  # List of scalers to be used in the tuning process.
      models,  # List of machine learning models to be used in the tuning process.
      fsTechs,  # List of feature selection techniques to be used in the tuning process.
      fsRatios,  # List of feature selection ratios to be used in the tuning process.
      dataBalanceTechniques,  # List of data balancing techniques to be used in the tuning process.
      outliersTechniques,  # List of outlier detection techniques to be used in the tuning process.
      datasetFilename,  # Name of the dataset file.
      storageFolderPath,  # Path to the folder where results will be stored.
      testFilename,  # Name of the test dataset file.
      testRatio=0.2,  # Ratio of the test data.
      contamination=0.05,  # Proportion of outliers in the data (for outlier detection techniques).
      numTrials=100,  # Number of trials for hyperparameter tuning.
      prefix="Optuna",  # Prefix for the study name and storage files.
      samplerTech="TPE",  # Sampler technique to be used for hyperparameter tuning.
      targetColumn="Class",  # Name of the target column in the dataset.
      dropFirstColumn=True,  # Whether to drop the first column (usually an index or ID).
      dropNAColumns=True,  # Whether to drop columns with any null values.
      encodeCategorical=True,  # Whether to encode categorical features (if any).
      saveFigures=True,  # Whether to save the performance plots.
      eps=1e-8,  # Small value to avoid division by zero (if needed).
      loadStudy=True,  # Whether to load an existing study if available.
      verbose=False,  # Whether to print verbose output.
  ):
    r'''
    Initializes the OptunaTuningClassification class with the provided hyperparameters.

    Parameters:
      scalers (list): List of scalers to be used in the tuning process.
      models (list): List of machine learning models to be used in the tuning process.
      fsTechs (list): List of feature selection techniques to be used in the tuning process.
      fsRatios (list): List of feature selection ratios to be used in the tuning process.
      dataBalanceTechniques (list): List of data balancing techniques to be used in the tuning process.
      outliersTechniques (list): List of outlier detection techniques to be used in the tuning process.
      baseDir (str): Base directory where the dataset is stored.
      datasetFilename (str): Name of the dataset file to be used for training.
      storageFolderPath (str): Path to the folder where results will be stored.
      testFilename (str): Name of the test dataset file to be used for evaluation.
      testRatio (float): Ratio of the test data to be used in the classification.
      contamination (float): Proportion of outliers in the data (for outlier detection techniques).
      numTrials (int): Number of trials for hyperparameter tuning.
      prefix (str): Prefix for the study name and storage files.
      samplerTech (str): Sampler technique to be used for hyperparameter tuning. Options are "TPE", "Random", "NSGAIISampler", or "CmaEs".
      targetColumn (str): Name of the target column in the dataset.
      dropFirstColumn (bool): Whether to drop the first column (usually an index or ID).
      dropNAColumns (bool): Whether to drop columns with any null values.
      encodeCategorical (bool): Whether to encode categorical features (if any).
      saveFigures (bool): Whether to save the performance plots.
      eps (float): Small value to avoid division by zero (if needed).
      loadStudy (bool): Whether to load an existing study if available.
      verbose (bool): Whether to print verbose output.
    '''

    if (not scalers or len(scalers) <= 0):
      raise ValueError("The list of scalers cannot be empty or None.")
    if (not models or len(models) <= 0):
      raise ValueError("The list of models cannot be empty or None.")
    if (not fsTechs or len(fsTechs) <= 0):
      raise ValueError("The list of feature selection techniques cannot be empty or None.")
    if (not fsRatios or len(fsRatios) <= 0):
      raise ValueError("The list of feature selection ratios cannot be empty or None.")
    if (not dataBalanceTechniques or len(dataBalanceTechniques) <= 0):
      raise ValueError("The list of data balancing techniques cannot be empty or None.")
    if (not outliersTechniques or len(outliersTechniques) <= 0):
      raise ValueError("The list of outlier detection techniques cannot be empty or None.")

    if (not baseDir or len(baseDir) <= 0):
      raise ValueError("The base directory cannot be empty or None.")
    if (not datasetFilename or len(datasetFilename) <= 0):
      raise ValueError("The dataset filename cannot be empty or None.")
    if (not storageFolderPath or len(storageFolderPath) <= 0):
      raise ValueError("The storage folder path cannot be empty or None.")

    if (not os.path.exists(os.path.join(baseDir, datasetFilename))):
      raise FileNotFoundError(f"The dataset file '{datasetFilename}' does not exist in the base directory '{baseDir}'.")
    if (testFilename and not os.path.exists(os.path.join(baseDir, testFilename))):
      raise FileNotFoundError(
        f"The test dataset file '{testFilename}' does not exist in the base directory '{baseDir}'."
      )

    self.scalers = scalers  # List of scalers to be used in the tuning process.
    self.models = models  # List of machine learning models to be used in the tuning process.
    self.fsTechs = fsTechs  # List of feature selection techniques to be used in the tuning process.
    self.fsRatios = fsRatios  # List of feature selection ratios to be used in the tuning process.
    self.dataBalanceTechniques = dataBalanceTechniques  # List of data balancing techniques to be used in the tuning process.
    self.outliersTechniques = outliersTechniques  # List of outlier detection techniques to be used in the tuning process.
    self.baseDir = baseDir  # Base directory where the dataset is stored.
    self.datasetFilename = datasetFilename  # Name of the dataset file.
    self.storageFolderPath = storageFolderPath  # Path to the folder where results will be stored.
    self.testFilename = testFilename  # Name of the test dataset file.
    self.contamination = contamination  # Proportion of outliers in the data (for outlier detection techniques).
    self.dropNAColumns = dropNAColumns  # Whether to drop columns with any null values.
    self.encodeCategorical = encodeCategorical  # Whether to encode categorical features (if any).
    self.saveFigures = saveFigures  # Whether to save the performance plots.
    self.eps = eps  # Small value to avoid division by zero (if needed).
    self.loadStudy = loadStudy  # Whether to load an existing study if available.
    self.verbose = verbose  # Whether to print verbose output.

    # Path to the test dataset file, if provided.
    if (testFilename is not None):
      self.testFilePath = os.path.join(self.baseDir, testFilename)
    else:
      self.testFilePath = None

    self.testRatio = testRatio  # Ratio of the test data to be used in the classification.
    self.numTrials = numTrials  # Number of trials for hyperparameter tuning.
    self.prefix = prefix  # Prefix for the study name and storage files.
    self.targetColumn = targetColumn  # Name of the target column in the dataset.
    self.dropFirstColumn = dropFirstColumn  # Whether to drop the first column (usually an index or ID).

    if (samplerTech == "TPE"):
      self.sampler = optuna.samplers.TPESampler(
        seed=np.random.randint(0, 10000),
      )
    elif (samplerTech == "Random"):
      self.sampler = optuna.samplers.RandomSampler(
        seed=np.random.randint(0, 10000),
      )
    elif (samplerTech == "CmaEs"):
      self.sampler = optuna.samplers.CmaEsSampler(
        seed=np.random.randint(0, 10000),
      )
    elif (samplerTech == "NSGAIISampler"):
      self.sampler = optuna.samplers.NSGAIISampler(
        seed=np.random.randint(0, 10000),
        population_size=100,  # Population size for NSGA-II.
        crossover_prob=0.9,  # Crossover probability.
        mutation_prob=0.1,  # Mutation probability.
      )
    else:
      raise ValueError(
        f"Unknown sampler technique: {samplerTech}. "
        f"Please choose from 'TPE', 'Random', 'NSGAIISampler', or 'CmaEs'."
      )

    # Apply random shuffling to the lists to ensure randomness in the selection of hyperparameters.
    np.random.shuffle(self.scalers)  # Shuffle the scalers for randomness.
    np.random.shuffle(self.models)  # Shuffle the models for randomness.
    np.random.shuffle(self.fsTechs)  # Shuffle the feature selection techniques for randomness.
    np.random.shuffle(self.fsRatios)  # Shuffle the feature ratios for randomness.
    np.random.shuffle(self.dataBalanceTechniques)  # Shuffle the data sampling techniques for randomness.
    np.random.shuffle(self.outliersTechniques)  # Shuffle the outlier detection techniques for randomness.

    # History variable to store the history of the trials.
    self.history = []

    # Initialize the study and best parameters.
    self.study = None  # To store the Optuna study object.
    self.bestParams = None  # To store the best hyperparameters found by the study.
    self.bestValue = None  # To store the best value found by the study.

    # Ensure the storage folder exists.
    os.makedirs(self.storageFolderPath, exist_ok=True)

    if (self.verbose):
      print("OptunaTuningClassification initialized.", flush=True)

  def ObjectiveFunction(
      self,
      trial,  # Optuna trial object.
  ):
    r'''
    Objective function for Optuna to optimize hyperparameters for machine learning classification.
    This function performs machine learning classification using the specified hyperparameters,
    saves the results, and returns the weighted average of the metrics.

    Parameters:
      optuna.Trial: The Optuna trial object containing the hyperparameters to be optimized.

    Returns:
      float: The weighted average of the metrics obtained from the machine learning classification.
    '''

    # Get the parameters for the machine learning classification.
    modelName = trial.suggest_categorical("Model", self.models)
    scalerName = trial.suggest_categorical("Scaler", self.scalers)
    fsTech = trial.suggest_categorical("FS Tech", self.fsTechs)
    fsRatio = trial.suggest_categorical("FS Ratio", self.fsRatios)
    dataBalanceTech = trial.suggest_categorical("DB Tech", self.dataBalanceTechniques)
    outliersTech = trial.suggest_categorical("Outliers Tech", self.outliersTechniques)

    try:
      # Call the function to perform machine learning classification.
      metrics, pltObject, objects = MachineLearningClassification(
        os.path.join(self.baseDir, self.datasetFilename),  # Path to the dataset file.
        scalerName,  # Name of the scaler to be used.
        modelName,  # Name of the machine learning model to be used.
        fsTech,  # Feature selection technique to be applied.
        fsTechRatio=fsRatio,  # Ratio of features to be selected.
        dataBalanceTech=dataBalanceTech,  # Data balancing technique to be applied.
        outlierTech=outliersTech,  # Outlier detection technique to be applied.
        contamination=self.contamination,  # Proportion of outliers in the data (for outlier detection techniques).
        testRatio=self.testRatio,  # Ratio of the test data.
        testFilePath=self.testFilePath,  # Path to the test dataset file.
        targetColumn=self.targetColumn,  # Name of the target column in the dataset.
        dropFirstColumn=self.dropFirstColumn,  # Whether to drop the first column (usually an index or ID).
        dropNAColumns=self.dropNAColumns,  # Whether to drop columns with any null values.
        encodeCategorical=self.encodeCategorical,  # Whether to encode categorical features (if any).
        eps=self.eps,  # Small value to avoid division by zero (if needed).
      )

      # Create a pattern for the filename based on model name, scaler name, feature selection technique, and ratio.
      pattern = (
        f"{modelName}_{scalerName}_{fsTech}_"
        f"{fsRatio if (fsTech is not None) else None}_{dataBalanceTech}_"
        f"{outliersTech if (outliersTech is not None) else None}"
      )

      if (self.saveFigures):
        # Added to check if the plot object is not None before saving.
        if (pltObject is not None):
          # Save the confusion matrix plot with a specific filename as a PNG image.
          pltObject.figure.savefig(
            os.path.join(self.storageFolderPath, f"{pattern}_CM.png"),
            bbox_inches="tight",  # Adjust the bounding box to fit the plot.
            dpi=720,  # Set the DPI for the saved image.
          )

          # pltObject.figure.show()  # Display the confusion matrix plot.
          # pltObject.figure.clf()  # Clear the figure to free up memory.
          plt.close(pltObject.figure)  # Close the plot to free up memory.

      # Added to check if the objects are not None before saving.
      if (objects is not None):
        # Save the trained model and scaler objects using pickle.
        with open(
            os.path.join(self.storageFolderPath, f"{pattern}.p"),
            "wb",  # Open the file in write-binary mode.
        ) as f:
          pickle.dump(objects, f)  # Save the model and scaler objects.

      # Append the model name and scaler name to the metrics dictionary.
      self.history.append(
        {
          "Model"                      : modelName,  # Name of the machine learning model.
          "Scaler"                     : scalerName,  # Name of the scaler used for preprocessing.
          "Feature Selection Technique": fsTech,  # Feature selection technique used.
          # Ratio of features selected, or "N/A" if no feature selection was applied.
          "Feature Selection Ratio"    : fsRatio if (fsTech is not None) else None,
          "Data Balancing Technique"   : dataBalanceTech,  # Data balancing technique used.
          "Outlier Detection Technique": outliersTech if (outliersTech is not None) else None,
          **metrics,
        }
      )

      if (self.verbose):
        print(
          f"Trial {trial.number}: Model={modelName}, Scaler={scalerName}, "
          f"FS Tech={fsTech}, FS Ratio={fsRatio if (fsTech is not None) else None}, "
          f"DB Tech={dataBalanceTech}, Outliers Tech={outliersTech if (outliersTech is not None) else None} "
          f"=> Weighted Average={metrics['Weighted Average']}",
          flush=True
        )

      # Return the weighted average of the metrics.
      return metrics["Weighted Average"]
    except Exception as e:
      # Uncomment the following line to print the error.
      print(f"\nError: {e}", flush=True)
      return 0.0

  def Tune(self):
    '''
    Tunes the hyperparameters using Optuna for machine learning classification.
    This function creates an Optuna study, optimizes the objective function,
    and saves the results including the best hyperparameters, trials history, and performance metrics.
    It also saves the study object for future reference.
    '''

    # Create the study object.
    self.study = optuna.create_study(
      direction="maximize",  # To maximize the objective function.
      study_name=f"{self.prefix}_Study",  # The study name.
      storage=f"sqlite:///{self.storageFolderPath}/{self.prefix}_Study.db",  # The database file.
      load_if_exists=self.loadStudy,  # To load the study if it exists.
      # https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
      sampler=self.sampler,  # Setting the sampler.
    )

    if (self.verbose):
      print(f"Study Name: {self.study.study_name}", flush=True)
      print(f"Number of Current/Completed Trials: {len(self.study.trials)}", flush=True)

    # Create the objective function with the arguments.
    objectiveFunction = lambda trial: self.ObjectiveFunction(trial)

    # Optimize the objective function.
    self.study.optimize(
      objectiveFunction,  # Objective function.
      n_trials=self.numTrials,  # Number of trials.
      n_jobs=1,  # To use all available CPUs for parallel execution. 1: Use one CPU.
      show_progress_bar=self.verbose,  # To show the progress bar.
    )

    if (self.verbose):
      print("Number of finished trials: ", len(self.study.trials), flush=True)
      print("Best trial:", flush=True)
      trial = self.study.best_trial
      print("  Value: ", trial.value, flush=True)
      print("  Params: ", flush=True)
      for key, value in trial.params.items():
        print(f"    {key}: {value}", flush=True)

    # Save the performance metrics in a CSV file for future reference.
    historyDF = pd.DataFrame(self.history)
    historyDF.to_csv(
      os.path.join(self.storageFolderPath, "Metrics History.csv"),
      index=False,
    )

    # Store the trials in a dataframe.
    trials = self.study.trials_dataframe()
    trials.to_csv(
      os.path.join(self.storageFolderPath, "Optuna Trials History.csv"),
      index=False,
    )

    # Get the best hyperparameters and the best value.
    self.bestParams = self.study.best_params
    self.bestValue = self.study.best_value
    print("Best Parameters:", self.bestParams, flush=True)
    print("Best Value:", self.bestValue, flush=True)

    # Save the best hyperparameters to a CSV file.
    bestParamsDF = pd.DataFrame(self.bestParams, index=[0])
    bestParamsDF.to_csv(
      os.path.join(self.storageFolderPath, "Optuna Best Params.csv"),
      index=False,
    )

    # Save the study object.
    with open(os.path.join(self.storageFolderPath, "Optuna Study.p"), "wb") as file:
      pickle.dump(self.study, file)

    if (self.verbose):
      print("Study saved successfully.", flush=True)

  def GetStudy(self):
    '''
    Returns the Optuna study object.

    Returns:
      optuna.study.Study: The Optuna study object.
    '''

    return self.study

  def GetBestParams(self):
    '''
    Returns the best hyperparameters found by the Optuna study.

    Returns:
      dict: A dictionary containing the best hyperparameters.
    '''

    return self.bestParams

  def GetBestValue(self):
    '''
    Returns the best value found by the Optuna study.

    Returns:
      float: The best value found by the Optuna study.
    '''

    return self.bestValue

  def LoadStudy(self, studyFilePath):
    '''
    Loads the Optuna study from a file.

    Parameters:
      studyFilePath (str): The path to the study file.

    Returns:
      optuna.study.Study: The loaded Optuna study object.
    '''

    with open(studyFilePath, "rb") as file:
      self.study = pickle.load(file)
      self.bestParams = self.study.best_params
      self.bestValue = self.study.best_value
    print("Study loaded successfully.", flush=True)
    return self.study


def OptunaTuningClassificationTesting(
    datasetFilePath,
    experimentPath,
    storageDir,
    dpi=720,
    T=500,
    plotCounterfactualOutcomes=False,
    numberOfTopExperiments=1,
    sortByMetric="Weighted Average",
):
  r'''
  Run inference using a saved preprocessing + model objects bundle.

  Parameters:
    datasetFilePath (str): Path to the dataset file for inference.
    experimentPath (str): Path to the experiment folder containing the saved model and preprocessing objects.
    storageDir (str): Directory where the performance plots and results will be saved.
    dpi (int, optional): DPI for saving the performance plots. Default is 720.
    T (int, optional): Number of Monte Carlo samples for uncertainty estimation (if applicable). Default is 500.
    plotCounterfactualOutcomes (bool, optional): Whether to plot counterfactual outcomes (if applicable). Default is False.
    numberOfTopExperiments (int, optional): Number of top experiments to test based on the best parameters. Default is 1.
    sortByMetric (str, optional): The metric by which to sort the experiments when selecting the top experiments. Default is "Weighted Average".
  '''

  from sklearn.metrics import confusion_matrix, classification_report
  from HMB.Utils import ReadPickleFile
  from HMB.PerformanceMetrics import (
    PlotConfusionMatrix, CalculatePerformanceMetrics, PlotErrorAnalysis,
    PlotROCAUCCurve, PlotPRCCurve, PlotClasswisePRFBar, PlotErrorMatrix,
    PlotPredictionConfidenceHistogram, ComputeECEPlotReliability,
    PlotClassificationResiduals, PlotFeatureImportance, RiskCoverageCurve,
    SampleMonteCarloDirichletFromProbs, ComputeMonteCarloUncertaintyMeasures,
    PlotCalibrationCurve, PlotTopKAccuracyCurve, PlotCounterfactualOutcomes,
  )
  from HMB.Initializations import UpdateMatplotlibSettings

  UpdateMatplotlibSettings()

  # Load the best parameters from a previous training run for the current file.
  bestParamsPath = os.path.join(experimentPath, "Metrics History.csv")
  if (os.path.exists(bestParamsPath)):
    bestParams = pd.read_csv(bestParamsPath)
    bestParams.fillna("None", inplace=True)
    if (numberOfTopExperiments > 1):
      bestParams = bestParams.sort_values(by=sortByMetric, ascending=False)
      # Group by the model column and find the top-1 in each group, then reset the index for the resultant DataFrame.
      bestParams = bestParams.groupby("Model").head(1).reset_index(drop=True)
      # Truncate the DataFrame to the specified number of top experiments if needed.
      if (numberOfTopExperiments > 0 and len(bestParams) > numberOfTopExperiments):
        bestParams = bestParams.head(numberOfTopExperiments)
      print(f"Top {numberOfTopExperiments} experiments based on {sortByMetric} for each model:")
      print(bestParams)
      # Store the resultant DF in a CSV file for reference.
      bestParams.to_csv(
        storageDir + f"/Top_{numberOfTopExperiments}_Experiments_Based_on_{sortByMetric}.csv",
        index=False
      )
      print(
        f"Top {numberOfTopExperiments} experiments saved to: "
        f"{storageDir}/Top_{numberOfTopExperiments}_Experiments_Based_on_{sortByMetric}.csv"
      )
  else:
    raise FileNotFoundError(f"Best parameters file not found at path: {bestParamsPath}")

  finalHistory = []
  for i in range(len(bestParams)):
    record = bestParams.iloc[i]
    scalerName = record["Scaler"]
    modelName = record["Model"]
    fsTechName = record["Feature Selection Technique"]
    fsRatioName = int(record["Feature Selection Ratio"]) if (fsTechName != "None") else "None"
    dbTechName = record["Data Balancing Technique"]
    outliersTechName = record["Outlier Detection Technique"]

    newStorageDir = os.path.join(storageDir, modelName)
    os.makedirs(newStorageDir, exist_ok=True)

    print("Testing the best model with the following parameters:")
    print(
      f"Scaler: {scalerName}, Model: {modelName}, FS Tech: {fsTechName}, FS Ratio: {fsRatioName}, "
      f"DB Tech: {dbTechName}, Outliers Tech: {outliersTechName}"
    )
    pickleFileName = f"{modelName}_{scalerName}_{fsTechName}_{fsRatioName}_{dbTechName}_{outliersTechName}.p"
    pickleFilePath = os.path.join(experimentPath, pickleFileName)
    if (not os.path.exists(pickleFilePath)):
      raise FileNotFoundError(f"Pickle file not found for the best model at path: {pickleFilePath}")
    # Copy the pickle file to the new storage directory for reference.
    newPickleFilePath = os.path.join(newStorageDir, pickleFileName)
    if (pickleFilePath != newPickleFilePath):
      shutil.copy(pickleFilePath, newPickleFilePath)
      print(f"Copied pickle file to: {newPickleFilePath}")

    # Load the best model from the pickle file.
    pickleData = ReadPickleFile(pickleFilePath)
    print(f"Loaded from pickle file: {pickleFilePath}")
    print(f"Pickle keys:")
    for key in pickleData.keys():
      print(f"  - {key}")

    model = pickleData.get("Model", None)
    if (model is None):
      raise KeyError(f"'Model' key not found in the pickle file: {pickleFilePath}")
    selectedFeatures = pickleData.get("SelectedFeatures", None)
    if (selectedFeatures is None):
      raise KeyError(f"'SelectedFeatures' key not found in the pickle file: {pickleFilePath}")
    labelEncoder = pickleData.get("LabelEncoder", None)
    if (labelEncoder is None):
      raise KeyError(f"'LabelEncoder' key not found in the pickle file: {pickleFilePath}")
    featuresEncoders = pickleData.get("FeaturesEncoders", None)
    if (featuresEncoders is None):
      raise KeyError(f"'FeaturesEncoders' key not found in the pickle file: {pickleFilePath}")
    scaler = pickleData.get("Scaler", None)
    if (scaler is None and not scalerName in ["None", None]):
      raise KeyError(f"'Scaler' key not found in the pickle file: {pickleFilePath}")
    configurations = pickleData.get("Configurations", None)
    if (configurations is None):
      raise KeyError(f"'Configurations' key not found in the pickle file: {pickleFilePath}")
    currentColumns = pickleData.get("CurrentColumns", None)
    if (currentColumns is None):
      raise KeyError(f"'CurrentColumns' key not found in the pickle file: {pickleFilePath}")
    featureSelector = pickleData.get("FeatureSelector", None)
    if (featureSelector is None and not fsTechName in ["None", None] and not fsRatioName in ["None", None, 100]):
      raise KeyError(f"'FeatureSelector' key not found in the pickle file: {pickleFilePath}")

    dropFirstColumn = configurations.get("dropFirstColumn", None)
    if (dropFirstColumn is None):
      raise KeyError(f"'dropFirstColumn' key not found in the configurations in the pickle file: {pickleFilePath}")
    targetColumn = configurations.get("targetColumn", None)
    if (targetColumn is None):
      raise KeyError(f"'targetColumn' key not found in the configurations in the pickle file: {pickleFilePath}")
    dropNAColumns = configurations.get("dropNAColumns", None)
    if (dropNAColumns is None):
      raise KeyError(f"'dropNAColumns' key not found in the configurations in the pickle file: {pickleFilePath}")

    # Load the dataset.
    if (not os.path.exists(datasetFilePath)):
      raise FileNotFoundError(f"Dataset file not found at path: {datasetFilePath}")
    # Read the CSV file into a pandas DataFrame.
    data = pd.read_csv(datasetFilePath)

    # Check if dropFirstColumn is True, then drop the first column.
    if (dropFirstColumn):
      # Drop the first column if it is not the target column.
      if (data.columns[0] != targetColumn):
        data = data.drop(data.columns[0], axis=1)

    # Drop columns with any null values if dropNAColumns is True.
    if (dropNAColumns):
      # Drop empty columns from the DataFrame.
      # Updated from "all" to "any" to drop columns with any null values.
      # axis=1: means columns, how="any" means drop if any value is null.
      data = data.dropna(axis=1, how="any")

      # Drop rows with null or empty values from the DataFrame.
      # axis=0: means rows, how="any" means drop if any value is null.
      data = data.dropna(axis=0, how="any")

    # Features (X) are all columns except the "Class" column.
    X = data.drop(targetColumn, axis=1)

    # Target (y) is the "Class" column.
    y = data[targetColumn]

    # Encode the target labels using the loaded label encoder.
    y = labelEncoder.transform(y)

    # Encode the features using the loaded features encoders if they exist.
    if (featuresEncoders is not None):
      categoricalCols = X.select_dtypes(include=["object", "category"]).columns
      for col in categoricalCols:
        if (col in featuresEncoders):
          encoder = featuresEncoders[col]
          X[col] = encoder.transform(X[col])
        else:
          raise KeyError(f"Encoder for feature column '{col}' not found in the loaded features encoders.")

    dataDf = pd.DataFrame(data, columns=currentColumns)
    if (scaler is not None):
      # Transform the data using the fitted scaler.
      dataDf = scaler.transform(dataDf)
      dataDf = pd.DataFrame(dataDf, columns=currentColumns)

    if (featureSelector is not None):
      if (not hasattr(featureSelector, "transform")):
        # Use the selected features list to select the features from the dataset.
        missingFeatures = [feat for feat in selectedFeatures if feat not in dataDf.columns]
        if (missingFeatures):
          raise ValueError(f"The following selected features are missing from the dataset: {missingFeatures}")
        dataDf = dataDf[selectedFeatures]
      else:
        # Select features using the fitted feature selector.
        dataDf = featureSelector.transform(dataDf)
      dataDf = pd.DataFrame(dataDf, columns=selectedFeatures)

    # Ensure that the selected features are in the same order as the selectedFeatures list if it exists.
    if (selectedFeatures is not None):
      missingFeatures = [feat for feat in selectedFeatures if feat not in dataDf.columns]
      if (missingFeatures):
        raise ValueError(f"The following selected features are missing from the dataset: {missingFeatures}")
      dataDf = dataDf[selectedFeatures]

    # Apply the loaded model to the processed dataset and evaluate the performance.
    yTrue = y
    yPred = model.predict(dataDf)
    yPredProb = model.predict_proba(dataDf) if (hasattr(model, "predict_proba")) else None
    yPredConfidences = np.max(yPredProb, axis=1) if (yPredProb is not None) else None
    if (labelEncoder is not None):
      yTrueLabels = labelEncoder.inverse_transform(yTrue)
      yPredLabels = labelEncoder.inverse_transform(yPred)
      classLabels = labelEncoder.classes_.tolist()
    else:
      yTrueLabels = yTrue
      yPredLabels = yPred
      classLabels = np.unique(yTrue).tolist()

    # Store evaluation results as DataFrame and save to CSV.
    evalResults = {
      "yTrue"           : yTrue.tolist(),
      "yPred"           : yPred.tolist(),
      "yPredProb"       : yPredProb.tolist() if (yPredProb is not None) else None,
      "yPredConfidences": yPredConfidences.tolist() if (yPredConfidences is not None) else None,
      "yTrueLabels"     : yTrueLabels.tolist(),
      "yPredLabels"     : yPredLabels.tolist(),
    }
    evalResultsDf = pd.DataFrame(evalResults)
    evalResultsFilePath = os.path.join(newStorageDir, f"EvaluationResults.csv")
    evalResultsDf.to_csv(evalResultsFilePath, index=False)

    cm = confusion_matrix(yTrue, yPred)
    cmPath = os.path.join(newStorageDir, f"ConfusionMatrix.pdf")
    PlotConfusionMatrix(
      cm,  # Confusion matrix (2D list or numpy array).
      classLabels,  # List of class labels.
      normalize=False,  # Whether to normalize the confusion matrix.
      roundDigits=3,  # Number of decimal places to round normalized values.
      title="Confusion Matrix",  # Title of the plot.
      cmap="Blues",  # Colormap for the heatmap.
      display=False,  # Whether to display the plot.
      save=True,  # Whether to save the plot.
      fileName=cmPath,  # File name to save the plot.
      fontSize=15,  # Font size for labels and annotations.
      annotate=True,  # Whether to annotate cells with values.
      figSize=(8, 8),  # Figure size in inches.
      colorbar=True,  # Whether to show colorbar.
      returnFig=False,  # Whether to return the figure object.
      dpi=dpi,  # DPI for saving the figure.
    )
    print("\nClassification Report:")
    print(classification_report(yTrue, yPred, target_names=classLabels))

    pm = CalculatePerformanceMetrics(
      confMatrix=cm,
      eps=1e-10,  # Small value to avoid division by zero.
      addWeightedAverage=True,  # Whether to include weighted averages in the output.
      addPerClass=True,  # Whether to include per-class metrics in the output.
    )
    for k in pm.keys():
      # If a list of values is present for a metric,
      # convert it to a string for better display in the CSV file.
      if (
          isinstance(pm[k], list) or
          # isinstance(pm[k], np.ndarray) or
          # isinstance(pm[k], pd.Series) or
          isinstance(pm[k], dict) or
          isinstance(pm[k], tuple)
      ):
        pm[k] = str(pm[k])
    pmDf = pd.DataFrame(pm)
    pmFilePath = os.path.join(newStorageDir, f"PerformanceMetrics.csv")
    pmDf.to_csv(pmFilePath, index=False)
    for key, value in pm.items():
      print(f"{key}: {value}")

    # Append the performance metrics to the final history list for all top experiments.
    finalHistory.append(
      {
        "Model"                      : modelName,
        "Scaler"                     : scalerName,
        "Feature Selection Technique": fsTechName,
        "Feature Selection Ratio"    : fsRatioName,
        "Data Balancing Technique"   : dbTechName,
        "Outlier Detection Technique": outliersTechName,
        **pm,
      }
    )

    rocPath = os.path.join(newStorageDir, f"ROCCurve.pdf")
    PlotROCAUCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb if (yPredProb is not None) else yPred,  # Predicted probabilities or labels.
      classLabels,  # List of class names.
      areProbabilities=True if (yPredProb is not None) else False,  # Whether yPred are probabilities.
      title="ROC Curve & AUC",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap=None,  # Colormap for ROC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=rocPath,  # File name.
      fontSize=16,  # Font size.
      plotDiagonal=True,  # Plot diagonal reference line.
      annotateAUC=True,  # Annotate AUC value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    prcPath = os.path.join(newStorageDir, f"PRCCurve.pdf")
    PlotPRCCurve(
      yTrue,  # True labels (one-hot or binary).
      yPredProb if (yPredProb is not None) else yPred,  # Predicted probabilities or labels.
      classLabels,  # List of class names.
      areProbabilities=True if (yPredProb is not None) else False,  # Whether yPred are probabilities.
      title="PRC Curve",  # Plot title.
      figSize=(5, 5),  # Figure size.
      cmap=None,  # Colormap for PRC curves.
      display=False,  # Display the plot.
      save=True,  # Save the plot.
      fileName=prcPath,  # File name.
      fontSize=16,  # Font size.
      annotateAvg=True,  # Annotate average precision value on plot.
      showLegend=True,  # Show legend.
      returnFig=False,  # Return figure object.
      dpi=dpi,  # DPI for saving the figure.
    )

    clsWisePRFPath = os.path.join(newStorageDir, f"ClasswisePRFBarPlot.pdf")
    PlotClasswisePRFBar(
      cm,
      classNames=classLabels,
      fontSize=15,
      figsize=(8, 5),
      display=False,
      save=True,
      fileName=clsWisePRFPath,
      dpi=dpi,
      returnFig=False,
    )

    errorAnalysisPath = os.path.join(newStorageDir, f"ErrorAnalysis.pdf")
    PlotErrorAnalysis(
      yTrue,
      yPred,
      X=None,
      classNames=classLabels,
      maxExamples=5,
      fontSize=15,
      figsize=(12, 10),
      display=False,
      save=True,
      fileName=errorAnalysisPath,
      dpi=dpi,
      returnFig=False,
    )

    errorMatrixPath = os.path.join(newStorageDir, f"ErrorMatrix.pdf")
    PlotErrorMatrix(
      cm,
      classNames=classLabels,
      fontSize=15,
      figsize=(7, 6),
      display=False,
      save=True,
      fileName=errorMatrixPath,
      dpi=dpi,
      returnFig=False,
    )

    if (yPredProb is not None):
      predConfHistPath = os.path.join(newStorageDir, f"PredictionConfidenceHistogram.pdf")
      PlotPredictionConfidenceHistogram(
        yPredProb,
        yPred=yPred,
        fontSize=15,
        figsize=(8, 5),
        bins=20,
        display=False,
        save=True,
        fileName=predConfHistPath,
        dpi=dpi,
        returnFig=False,
      )

    clsResidualsPath = os.path.join(newStorageDir, f"ClassificationResiduals.pdf")
    PlotClassificationResiduals(
      yTrue,
      yPred,
      fontSize=15,
      figsize=(8, 5),
      display=False,
      save=True,
      fileName=clsResidualsPath,
      dpi=dpi,
      returnFig=False,
    )

    if (hasattr(model, "feature_importances_") or hasattr(model, "coef_")):
      featureImportancePath = os.path.join(newStorageDir, f"FeatureImportance.pdf")
      PlotFeatureImportance(
        model,
        featureNames=selectedFeatures,
        title="Feature Importance",
        fontSize=14,
        figsize=(8, 5),
        display=False,
        save=True,
        fileName=featureImportancePath,
        dpi=dpi,
        returnFig=False,
        topN=10,
      )

    if (yPredProb is not None):
      probsMC = SampleMonteCarloDirichletFromProbs(
        yPredProb,
        T=T,
        concentration=30.0
      )
      uncertaintyMeasures = ComputeMonteCarloUncertaintyMeasures(probsMC)
      confidences = uncertaintyMeasures["predictedConfidence"]
      predictions = uncertaintyMeasures["predictedIdx"]
      ecePath = os.path.join(newStorageDir, f"ECEReliabilityPlot.pdf")
      ece, binAcc, binConf, binCounts = ComputeECEPlotReliability(
        confidences,
        predictions,
        labels=yPred,
        nBins=5,
        title="ECE Example",
        fontSize=15,
        figSize=(6, 6),
        display=False,
        save=True,
        fileName=ecePath,
        dpi=dpi,
        returnFig=False,
        cmap="Blues",
        applyXYLimits=True,
      )
      print(f"ECE: {ece}")
      print(f"Bin Accuracies: {binAcc}")
      print(f"Bin Confidences: {binConf}")
      print(f"Bin Counts: {binCounts}")

      riskCoveragePath = os.path.join(newStorageDir, f"RiskCoverageCurve.pdf")
      correctness = (predictions == yPred).astype(int)
      RiskCoverageCurve(
        confidences,
        correctness,
        title="Risk-Coverage (Accuracy vs Coverage)",
        fontSize=15,
        figSize=(6, 6),
        display=False,
        save=True,
        fileName=riskCoveragePath,
        dpi=dpi,
        returnFig=False,
        color="blue",
      )

      topKAccPath = os.path.join(newStorageDir, f"TopKAccuracyCurve.pdf")
      PlotTopKAccuracyCurve(
        yPredProb,
        yPred,
        maxK=10,
        title="Top-k Accuracy Curve",
        figSize=(6, 6),
        save=True,
        fileName=topKAccPath,
        display=False,
        fontSize=15,
        returnFig=False,
        dpi=dpi,
        color="blue",
      )

      calibCurvePath = os.path.join(newStorageDir, f"CalibrationCurve.pdf")
      PlotCalibrationCurve(
        yPredProb,
        yPred,
        nBins=10,
        title="Calibration Curve",
        fontSize=15,
        figSize=(6, 6),
        display=False,
        save=True,
        fileName=calibCurvePath,
        dpi=dpi,
        returnFig=False,
        color="blue",
      )

    if (plotCounterfactualOutcomes):
      counterFolder = os.path.join(newStorageDir, "Counterfactual Outcomes")
      os.makedirs(counterFolder, exist_ok=True)
      for col in selectedFeatures:
        counterfactualOutcomesPath = os.path.join(counterFolder, f"{col}.pdf")
        PlotCounterfactualOutcomes(
          dataDf,
          model,
          treatmentCol=col,
          lowVal=None,
          highVal=None,
          classNames=classLabels,
          title="Counterfactual Outcome Distributions",
          save=True,
          fileName=counterfactualOutcomesPath,
          display=False,
          returnPreds=False,
          fontSize=15,
          dpi=dpi,
        )

  # Store the final history of all top experiments in a CSV file.
  finalHistoryDf = pd.DataFrame(finalHistory)
  finalHistoryFilePath = os.path.join(storageDir, f"FinalHistoryOfTop_{numberOfTopExperiments}_Experiments.csv")
  finalHistoryDf.to_csv(finalHistoryFilePath, index=False)
  print(f"\nFinal history of top {numberOfTopExperiments} experiments saved to: {finalHistoryFilePath}")

  # Store as latex table as well for easy inclusion in reports.
  finalHistoryLatexPath = os.path.join(storageDir, f"FinalHistoryOfTop_{numberOfTopExperiments}_Experiments.tex")
  with open(finalHistoryLatexPath, "w") as f:
    f.write(finalHistoryDf.to_latex(index=False))
  print(f"Final history of top {numberOfTopExperiments} experiments saved to: {finalHistoryLatexPath}")

  # Initialize variable that will hold the path to the found FinalHistory CSV (if any).
  # Find the CSV file that starts with "FinalHistory" in the "Testing Results" folder.
  finalHistoryFile = None
  # Search the testing results directory for a file that looks like FinalHistory*.csv.
  for el in os.listdir(storageDir):
    # Check each filename to see if it matches the expected prefix/suffix.
    if (el.startswith("FinalHistory") and el.endswith(".csv")):
      # If found, build the absolute path and stop searching.
      finalHistoryFile = os.path.join(storageDir, el)
      # Break out of the search loop once a matching file is found.
      break

  # If a FinalHistory CSV was found, load and sanitize it for the merged report.
  if (finalHistoryFile):
    # Load the CSV into a pandas DataFrame for processing.
    finalHistoryDf = pd.read_csv(finalHistoryFile)
    # Confirm successful load to the user via a printed message.
    print(f"\u2713 Loaded final history from: {finalHistoryFile}")

    # Replace any NaN values with empty strings to improve readability in the report.
    # Replace NaN values with empty strings for better display.
    finalHistoryDf.fillna("", inplace=True)

    # Identify metric columns to convert to percentages and round them.
    # Round numeric values to 4 decimal places for better readability.
    # This should be done only for columns with prefix "Weighted" in their names,
    # as they represent the main performance metrics;
    # other columns may contain non-numeric values or values that should not be rounded.
    numericCols = [col for col in finalHistoryDf.columns if col.startswith("Weighted")]

    # Convert weighted metrics to percentages and round to two decimal places.
    finalHistoryDf[numericCols] = (finalHistoryDf[numericCols] * 100).round(2)

    # Sort rows so the best-performing models (by Weighted Average) appear first.
    # Sort each group (based on the model) by the "Weighted Average" metric in descending order to
    # have the best-performing models at the top of the report.
    finalHistoryDf.sort_values(by="Weighted Average", ascending=False, inplace=True)

    # Narrow down columns to keep only relevant performance metrics (exclude many specifics).
    # Keep only the metrics with the "Weighted" prefix in their names for a more concise report.
    # To keep the initial columns, we can do this by removing the columns with the prefix "Macro",
    # "Class", and "Micro" while keeping the rest.
    columnsToKeep = [
      col
      for col in finalHistoryDf.columns
      if (
        not (
            col.startswith("Macro") or  # Exclude columns that start with "Macro".
            "Class" in col or  # Exclude columns that contain "Class".
            col.startswith("Micro") or  # Exclude columns that start with "Micro".
            col.startswith("Weights") or  # Exclude columns that start with "Weights".
            "Average" in col  # Exclude columns that contain "Average".
        )
      )
    ]

    # Apply the column filtering to keep only the selected columns.
    finalHistoryDf = finalHistoryDf[columnsToKeep]
    # Rename columns to remove the verbose "Weighted " prefix for clarity.
    # Remove the prefix "Weighted " from the column names for better readability.
    finalHistoryDf.columns = [col.replace("Weighted ", "") for col in finalHistoryDf.columns]
    # Drop confusion-matrix counts to focus the report on metrics of interest.
    # Drop the TP, TN, FP, FN columns to focus on the main performance metrics.
    columnsToDrop = [col for col in finalHistoryDf.columns if (col in ["TP", "TN", "FP", "FN"])]
    finalHistoryDf.drop(columns=columnsToDrop, inplace=True)

    # Store the cleaned final history DataFrame in a CSV file for reference.
    cleanedFinalHistoryFilePath = os.path.join(
      storageDir,
      f"Cleaned_Summary_of_Top_{numberOfTopExperiments}_Experiments.csv"
    )
    finalHistoryDf.to_csv(cleanedFinalHistoryFilePath, index=False)
    print(f"\u2713 Cleaned summary of top {numberOfTopExperiments} experiments saved to: {cleanedFinalHistoryFilePath}")
  else:
    # Inform the user if no final history file was found for a given system.
    print(f"\u2717 Final history file not found in: {testingResultsPath}")


def OptunaTuningClassificationTrials(
    datasetFilePath,
    experimentPath,
    storageDir,
    noOfTrials=10,
    dpi=720,
    T=500,
    plotCounterfactualOutcomes=False,
    numberOfTopExperiments=1,
    sortByMetric="Weighted Average",
):
  r'''
  Run multiple trials of inference using a saved preprocessing + model objects bundle with the best parameters
  from a previous tuning run.

  Parameters:
    datasetFilePath (str): Path to the dataset file for inference.
    experimentPath (str): Path to the experiment folder containing the saved model and preprocessing objects.
    storageDir (str): Directory where the performance plots and results will be saved.
    noOfTrials (int, optional): Number of trials to run. Default is 10.
    dpi (int, optional): DPI for saving the performance plots. Default is 720.
    T (int, optional): Number of Monte Carlo samples for uncertainty estimation (if applicable). Default is 500.
    plotCounterfactualOutcomes (bool, optional): Whether to plot counterfactual outcomes (if applicable). Default is False.
    numberOfTopExperiments (int, optional): Number of top experiments to test based on the best parameters. Default is 1.
    sortByMetric (str, optional): The metric to sort the experiments by when selecting the top experiments. Default is "Weighted Average".
  '''

  from HMB.Utils import ReadPickleFile
  from HMB.Initializations import DoRandomSeeding

  # Load the best parameters from a previous training run for the current file.
  bestParamsPath = os.path.join(experimentPath, "Metrics History.csv")
  if (os.path.exists(bestParamsPath)):
    bestParams = pd.read_csv(bestParamsPath)
    bestParams.fillna("None", inplace=True)
    if (numberOfTopExperiments > 1):
      bestParams = bestParams.sort_values(by=sortByMetric, ascending=False)
      # Group by the model column and find the top-1 in each group, then reset the index for the resultant DataFrame.
      bestParams = bestParams.groupby("Model").head(1).reset_index(drop=True)
      # Truncate the DataFrame to the specified number of top experiments if needed.
      if (numberOfTopExperiments > 0 and len(bestParams) > numberOfTopExperiments):
        bestParams = bestParams.head(numberOfTopExperiments)
      print(f"Top {numberOfTopExperiments} experiments based on {sortByMetric} for each model:")
      print(bestParams)
      # Store the resultant DF in a CSV file for reference.
      bestParams.to_csv(
        storageDir + f"/Top_{numberOfTopExperiments}_Experiments_Based_on_{sortByMetric}.csv",
        index=False
      )
      print(
        f"Top {numberOfTopExperiments} experiments saved to: "
        f"{storageDir}/Top_{numberOfTopExperiments}_Experiments_Based_on_{sortByMetric}.csv"
      )
  else:
    raise FileNotFoundError(f"Best parameters file not found at path: {bestParamsPath}")

  for i in range(len(bestParams)):
    record = bestParams.iloc[i]
    scalerName = record["Scaler"]
    modelName = record["Model"]
    fsTechName = record["Feature Selection Technique"]
    fsRatioName = int(record["Feature Selection Ratio"]) if (fsTechName != "None") else "None"
    dbTechName = record["Data Balancing Technique"]
    outliersTechName = record["Outlier Detection Technique"]

    newStorageDir = os.path.join(storageDir, modelName)
    os.makedirs(newStorageDir, exist_ok=True)

    print("Testing the best model with the following parameters:")

    print("Running trials with the best parameters...")
    print(
      f"Scaler: {scalerName}, Model: {modelName}, FS Tech: {fsTechName}, FS Ratio: {fsRatioName}, "
      f"DB Tech: {dbTechName}, Outliers Tech: {outliersTechName}"
    )
    pickleFileName = f"{modelName}_{scalerName}_{fsTechName}_{fsRatioName}_{dbTechName}_{outliersTechName}.p"
    pickleFilePath = os.path.join(experimentPath, pickleFileName)
    if (not os.path.exists(pickleFilePath)):
      raise FileNotFoundError(f"Pickle file not found for the best model at path: {pickleFilePath}")

    # Load the best model from the pickle file.
    pickleData = ReadPickleFile(pickleFilePath)
    print(f"Loaded from pickle file: {pickleFilePath}")

    print(f"\nPickle keys:")
    for key in pickleData.keys():
      print(f"  - {key}")

    configurations = pickleData.get("Configurations", None)
    if (configurations is None):
      raise KeyError(f"'Configurations' key not found in the pickle file: {pickleFilePath}")
    print("\nConfigurations in the pickle file:")
    for key in configurations.keys():
      print(f"  - {key}: {configurations[key]}")

    dropFirstColumn = configurations.get("dropFirstColumn", None)
    if (dropFirstColumn is None):
      raise KeyError(f"'dropFirstColumn' key not found in the configurations in the pickle file: {pickleFilePath}")
    targetColumn = configurations.get("targetColumn", None)
    if (targetColumn is None):
      raise KeyError(f"'targetColumn' key not found in the configurations in the pickle file: {pickleFilePath}")
    dropNAColumns = configurations.get("dropNAColumns", None)
    if (dropNAColumns is None):
      raise KeyError(f"'dropNAColumns' key not found in the configurations in the pickle file: {pickleFilePath}")
    encodeCategorical = configurations.get("encodeCategorical", None)
    if (encodeCategorical is None):
      raise KeyError(f"'encodeCategorical' key not found in the configurations in the pickle file: {pickleFilePath}")
    testRatio = configurations.get("testRatio", None)
    if (testRatio is None):
      raise KeyError(f"'testRatio' key not found in the configurations in the pickle file: {pickleFilePath}")
    eps = configurations.get("eps", None)
    if (eps is None):
      raise KeyError(f"'eps' key not found in the configurations in the pickle file: {pickleFilePath}")
    testFilePath = configurations.get("testFilePath", None)
    contamination = configurations.get("contamination", None)
    if (contamination is None):
      raise KeyError(f"'contamination' key not found in the configurations in the pickle file: {pickleFilePath}")

    for trial in range(noOfTrials):
      DoRandomSeeding()

      print(f"\nTrial {trial + 1}/{noOfTrials}", flush=True)
      trialStorageDir = os.path.join(newStorageDir, f"Trial_{trial + 1}")
      os.makedirs(trialStorageDir, exist_ok=True)

      # Copy the best parameters file to the trial storage directory for reference.
      trialBestParamsPath = os.path.join(trialStorageDir, "Optuna Best Params.csv")
      pd.DataFrame([record]).to_csv(trialBestParamsPath, index=False)

      # Run a new experiment with the best parameters for the current file and
      # save the results in a separate folder for each trial.
      # Call the function to perform machine learning classification.
      metrics, pltObject, objects = MachineLearningClassification(
        datasetFilePath,
        scalerName if (scalerName != "None") else None,
        modelName,
        fsTechName if (fsTechName != "None") else None,
        fsTechRatio=fsRatioName if (fsTechName != "None") else None,
        dataBalanceTech=dbTechName if (dbTechName != "None") else None,
        outlierTech=outliersTechName if (outliersTechName != "None") else None,
        contamination=contamination,
        testRatio=testRatio,
        testFilePath=testFilePath if (testFilePath is not None and os.path.exists(testFilePath)) else None,
        targetColumn=targetColumn,
        dropFirstColumn=dropFirstColumn,
        dropNAColumns=dropNAColumns,
        encodeCategorical=encodeCategorical,
        eps=eps,
      )

      # Create a pattern for the filename based on model name, scaler name, feature selection technique, and ratio.
      pattern = (
        f"{modelName}_{scalerName}_{fsTechName}_"
        f"{fsRatioName if (fsTechName is not None) else None}_{dbTechName}_"
        f"{outliersTechName if (outliersTechName is not None) else None}"
      )

      # Added to check if the plot object is not None before saving.
      if (pltObject is not None):
        # Save the confusion matrix plot with a specific filename as a PNG image.
        pltObject.figure.savefig(
          os.path.join(trialStorageDir, f"{pattern}_CM.png"),
          bbox_inches="tight",  # Adjust the bounding box to fit the plot.
          dpi=dpi,  # Set the DPI for the saved image.
        )

        # pltObject.figure.show()  # Display the confusion matrix plot.
        # pltObject.figure.clf()  # Clear the figure to free up memory.
        plt.close(pltObject.figure)  # Close the plot to free up memory.

      # Added to check if the objects are not None before saving.
      if (objects is not None):
        # Save the trained model and scaler objects using pickle.
        with open(
            os.path.join(trialStorageDir, f"{pattern}.p"),
            "wb",  # Open the file in write-binary mode.
        ) as f:
          pickle.dump(objects, f)  # Save the model and scaler objects.

      record = {
        "Model"                      : modelName,  # Name of the machine learning model.
        "Scaler"                     : scalerName,  # Name of the scaler used for preprocessing.
        "Feature Selection Technique": fsTechName,  # Feature selection technique used.
        # Ratio of features selected, or "N/A" if no feature selection was applied.
        "Feature Selection Ratio"    : fsRatioName if (fsTechName is not None) else None,
        "Data Balancing Technique"   : dbTechName,  # Data balancing technique used.
        "Outlier Detection Technique": outliersTechName if (outliersTechName is not None) else None,
        **metrics,
      }
      # Store the record as a CSV file for the current trial.
      recordDf = pd.DataFrame([record])  # Create a DataFrame from the record dictionary.
      recordFilePath = os.path.join(trialStorageDir, f"{pattern}_Metrics.csv")
      recordDf.to_csv(recordFilePath, index=False)  # Save the record as a CSV file.

      recordFilePath = os.path.join(trialStorageDir, f"Metrics History.csv")
      recordDf.to_csv(recordFilePath, index=False)  # Save the record as a CSV file.

      print(
        f"Trial {trial}: Model={modelName}, Scaler={scalerName}, "
        f"FS Tech={fsTechName}, FS Ratio={fsRatioName if (fsTechName is not None) else None}, "
        f"DB Tech={dbTechName}, Outliers Tech={outliersTechName if (outliersTechName is not None) else None} "
        f"=> Weighted Average={metrics['Weighted Average']}",
        flush=True
      )

      # Apply testing with the best parameters for the current file and save the results in a separate folder for each trial.
      testStorageDir = os.path.join(trialStorageDir, "Testing")
      os.makedirs(testStorageDir, exist_ok=True)
      OptunaTuningClassificationTesting(
        datasetFilePath=testFilePath if (
            testFilePath is not None and os.path.exists(testFilePath)
        ) else datasetFilePath,
        experimentPath=trialStorageDir,
        storageDir=testStorageDir,
        dpi=dpi,
        T=T,
        plotCounterfactualOutcomes=plotCounterfactualOutcomes,
        numberOfTopExperiments=1,
        sortByMetric=sortByMetric,
      )

  # Find subdirectories (per-model) inside the trial results path.
  foundModels = [
    el
    for el in os.listdir(storageDir)
    if (os.path.isdir(os.path.join(storageDir, el)))
  ]

  # Print which models were discovered with trial results.
  print(f"\u2713 Found the following models with trial results: {foundModels}")
  # Prepare a container to collect per-model trial histories.
  fileHistory = {}
  # Iterate each model folder to extract per-trial first-row summaries.
  for model in foundModels:
    # Print progress for the model being processed.
    print(f"\u2713 Processing trial results for model: {model}")
    # Path to the model's trial results directory.
    modelTrialResultsPath = os.path.join(storageDir, model)
    # List all trial subfolders/files present for this model.
    modelTrials = [el for el in os.listdir(modelTrialResultsPath)]
    # Report which trial result entries were found for this model.
    print(f"\u2713 Found the following trial result files for model {model}: {modelTrials}")
    # Container for the first-row summary of each trial.
    modelHistory = []
    # For each trial folder, load the benchmark FinalHistoryOfTop_1_Experiments.csv.
    for trialFile in modelTrials:
      # Path to the CSV file expected inside each trial's Testing folder.
      csvFile = os.path.join(modelTrialResultsPath, trialFile, "Testing", "FinalHistoryOfTop_1_Experiments.csv")
      if (not os.path.exists(csvFile)):
        print(f"\u2717 Expected trial history file not found: {csvFile}")
        continue  # Skip to the next trial if the expected file is missing.
      # Read the CSV into a DataFrame and take the first row as the trial summary.
      df = pd.read_csv(csvFile)
      firstRow = df.iloc[0]
      modelHistory.append(firstRow)
    # Indicate completion of extraction for the current model.
    print(f"\u2713 Extracted the first row of metrics history from each trial for model {model}.")
    # Store the per-model trial histories into the fileHistory mapping.
    fileHistory[model] = modelHistory
  # Prepare an empty DataFrame to accumulate all models' trial summaries.
  historyDf = pd.DataFrame()
  # Convert each model's list of first-row summaries into a DataFrame and concatenate.
  for model, history in fileHistory.items():
    modelDf = pd.DataFrame(history)
    modelDf["Model"] = model  # Add a column for the model name.
    # Concatenate the DataFrames for all models.
    historyDf = pd.concat([historyDf, modelDf], ignore_index=True)
  # Reorder columns so "Model" appears first for readability.
  # Reorder the columns to have "Model" as the first column.
  cols = historyDf.columns.tolist()
  cols = ["Model"] + [col for col in cols if (col != "Model")]
  historyDf = historyDf[cols]
  # Identify metric columns to convert to percentages and round them.
  # Round numeric values to 4 decimal places for better readability.
  # This should be done only for columns with prefix "Weighted" in their names,
  # as they represent the main performance metrics;
  # other columns may contain non-numeric values or values that should not be rounded.
  numericCols = [col for col in historyDf.columns if col.startswith("Weighted")]
  historyDf[numericCols] = (historyDf[numericCols] * 100).round(2)
  # Sort rows so the best-performing models (by Weighted Average) appear first.
  # Sort the DataFrame by the "Model" metric in descending order to have the best-performing models at the top of the report.
  historyDf.sort_values(by="Model", ascending=False, inplace=True)
  # Keep only the metrics with the "Weighted" prefix in their names for a more concise report.
  # To keep the initial columns, we can do this by removing the columns with the prefix "Macro", "Class", and "Micro" while keeping the rest.
  columnsToKeep = [
    col
    for col in historyDf.columns
    if (
      not (
          col.startswith("Macro") or  # Exclude columns that start with "Macro".
          "Class" in col or  # Exclude columns that contain "Class".
          col.startswith("Micro") or  # Exclude columns that start with "Micro".
          col.startswith("Weights") or  # Exclude columns that start with "Weights".
          "Average" in col  # Exclude columns that contain "Average".
      )
    )
  ]
  # Apply the column filtering to keep only the selected columns.
  historyDf = historyDf[columnsToKeep]
  # Drop the TP, TN, FP, FN columns to focus on the main performance metrics.
  columnsToDrop = [col for col in historyDf.columns if (col in ["TP", "TN", "FP", "FN"])]
  historyDf.drop(columns=columnsToDrop, inplace=True)
  # Identify non-numeric columns to exclude from rounding and percentage conversion; typically,
  # these would be the "Model" column and any other categorical identifiers.
  nonNumericCols = [col for col in historyDf.columns if (not col.startswith("Weighted") and col != "Model")]
  # Remove the prefix "Weighted " from the column names for better readability.
  historyDf.columns = [col.replace("Weighted ", "") for col in historyDf.columns]
  # Save the aggregated trial history for this file to CSV for downstream inspection.
  # Save the history DataFrame to a CSV file for comparison.
  historyReportFile = os.path.join(storageDir, "Merged_Summary.csv")
  historyDf.to_csv(historyReportFile, index=False)
  print(f"\u2713 Merged trials history summary saved to: {historyReportFile}")
  # Also write a LaTeX table version of the merged trials history.
  # Save as Latex table for better presentation in reports or papers.
  historyLatexFile = os.path.join(storageDir, "Merged_Summary.tex")
  # Number should be formatted to 4 decimal places in the LaTeX table for better readability.
  with open(historyLatexFile, "w") as f:
    # Write the history DataFrame to LaTeX with consistent numeric formatting.
    f.write(historyDf.to_latex(index=False, float_format="%.2f"))
  print(f"\u2713 Merged trials history summary saved as LaTeX table to: {historyLatexFile}")
  # print(f"\u2713 Merged Trials History Summary for file {file}:\n{historyDf}")
  # Get the mean and standard deviation of the performance metrics for each model and save to a summary CSV.
  # Drop non-numeric columns before aggregation; keeping only the "Model" column and the metric columns for mean/std calculation.
  historyDf = historyDf.drop(columns=nonNumericCols)  # Drop non-numeric columns before aggregation.
  # Store the history DataFrame to a CSV file for comparison.
  historyReportFile = os.path.join(storageDir, "Refined_Summary.csv")
  historyDf.to_csv(historyReportFile, index=False)
  print(f"\u2713 Merged trials history summary saved to: {historyReportFile}")
  summaryDf = historyDf.groupby("Model").agg(["mean", "std"])
  summaryReportFile = os.path.join(storageDir, "Statistics_Summary.csv")
  summaryDf.to_csv(summaryReportFile)
  print(f"\u2713 Trials history summary statistics saved to: {summaryReportFile}")
  # Also write a LaTeX table version of the summary statistics.
  summaryLatexFile = os.path.join(storageDir, "Statistics_Summary.tex")
  with open(summaryLatexFile, "w") as f:
    f.write(summaryDf.to_latex(float_format="%.2f"))
  print(f"\u2713 Trials history summary statistics saved as LaTeX table to: {summaryLatexFile}")


def OptunaTuningClassificationTrialsStatistics(
    trialResultsPath,
    statisticsStoragePath,
    dpi=720,
    plotMetricsIndividual=False,
    plotMetricsOverall=False,
):
  r'''
  Analyze the results from multiple trials of inference using a saved preprocessing + model objects bundle
  with the best parameters from a previous tuning run.

  Parameters:
    trialResultsPath (str): Path to the directory containing the results of multiple trials.
    statisticsStoragePath (str): Directory where the aggregated performance plots and statistics will be saved.
    dpi (int, optional): DPI for saving the performance plots. Default is 720.
    plotMetricsIndividual (bool, optional): Whether to plot performance metrics for individual trials. Default is False.
    plotMetricsOverall (bool, optional): Whether to plot aggregated performance metrics across all trials. Default is False.

  Returns:
    dict: A dictionary containing the aggregated performance metrics and results from all trials.
  '''

  from sklearn.metrics import confusion_matrix
  from HMB.PerformanceMetrics import CalculatePerformanceMetrics, PlotMultiTrialROCAUC, PlotMultiTrialPRCurve
  from HMB.StatisticalAnalysisHelper import ExtractDataFromSummaryFile, PlotMetrics, StatisticalAnalysis

  # Check first if the folder contains multiple trial result folders with the expected structure or just a single
  # trial result folder.
  # Determine if there are any CSVs beginning with "Top" indicating trial outputs.
  # Check if there is a CSV file that starts with the prefix "Top".
  isFound = len([
    el for el in os.listdir(trialResultsPath)
    if (el.startswith("Top") and el.endswith(".csv"))
  ]) > 0
  filesToProcess = {}
  if (isFound):
    # Inform the user that trial result files were found in the directory.
    print("\u2713 Found multiple models trial results file in:", trialResultsPath)
    # Find subdirectories (per-model) inside the trial results' path.
    foundModels = [
      el
      for el in os.listdir(trialResultsPath)
      if (os.path.isdir(os.path.join(trialResultsPath, el)))
    ]
    # Print which models were discovered with trial results.
    print(f"\u2713 Found the following models with trial results: {foundModels}")
    for record in foundModels:
      recordPath = os.path.join(trialResultsPath, record)
      if (os.path.isdir(recordPath)):
        trialFiles = [
          el
          for el in os.listdir(recordPath)
          if (el.startswith("Trial_") and os.path.isdir(os.path.join(recordPath, el)))
        ]
        if (len(trialFiles) == 0):
          print(
            f"\u26A0 No trial result folders found for model '{record}' (skipping). "
            f"Expected subdirectories named 'Trial_X'."
          )
        else:
          filesToProcess[record] = trialFiles
      else:
        print(f"\u26A0 Found unexpected file in trial results directory (skipping): {recordPath}")
  else:
    trialFiles = [
      el
      for el in os.listdir(trialResultsPath)
      if (el.startswith("Trial_") and os.path.isdir(os.path.join(trialResultsPath, el)))
    ]
    if (len(trialFiles) == 0):
      raise ValueError(
        f"No trial result folders found in the specified directory: {trialResultsPath}. "
        f"Ensure that the directory contains subdirectories named 'Trial_X' for each trial."
      )
    filesToProcess["Current"] = trialFiles

  if (len(filesToProcess) == 0):
    raise ValueError(
      f"No trial result folders found in the specified directory: {trialResultsPath}. "
      f"Ensure that the directory contains subdirectories named 'Trial_X' for each trial."
    )

  print(f"\u2713 Found the following trial files to process: {list(filesToProcess.keys())}")

  allHistory = {}
  for k, trialFiles in filesToProcess.items():
    print(f"\nProcessing trial results for: {k}")
    if (k == "Current"):
      newStorageDir = statisticsStoragePath
    else:
      newStorageDir = os.path.join(statisticsStoragePath, k)
    os.makedirs(newStorageDir, exist_ok=True)

    allYTrue = []
    allYPred = []
    classes = []
    kHistory = {}

    for trialFile in trialFiles:
      if (k == "Current"):
        csvPath = os.path.join(trialResultsPath, trialFile, "Testing", "EvaluationResults.csv")
      else:
        csvPath = os.path.join(trialResultsPath, k, trialFile, "Testing", "EvaluationResults.csv")

      if (not os.path.isfile(csvPath)):
        raise FileNotFoundError(
          f"Expected file not found: {csvPath}. Please ensure that the trial results are "
          f"generated correctly and the file structure is as expected."
        )

      # yTrue	yPred	yPredProb	yPredConfidences	yTrueLabels	yPredLabels.
      df = pd.read_csv(csvPath)
      yTrue = df["yTrue"].values
      yPred = df["yPred"].values
      try:
        yPredProb = df["yPredProb"].values
        # Parse the yPredProb column from string representation of list to actual list of floats.
        yPredProb = np.array([np.fromstring(probStr.strip("[]"), sep=",") for probStr in yPredProb])
      except:
        yPredProb = None

      yPredConfidences = df["yPredConfidences"].values
      yTrueLabels = df["yTrueLabels"].values
      yPredLabels = df["yPredLabels"].values
      classes = np.unique(yTrueLabels)
      cm = confusion_matrix(yTrue, yPred)
      metrics = CalculatePerformanceMetrics(cm, addWeightedAverage=True, eps=1e-8)
      kHistory[trialFile] = {
        "yTrue"           : yTrue,
        "yPred"           : yPred,
        "yPredProb"       : yPredProb,
        "yPredConfidences": yPredConfidences,
        "yTrueLabels"     : yTrueLabels,
        "yPredLabels"     : yPredLabels,
        "confusionMatrix" : cm,
        "metrics"         : metrics,
      }
      allYTrue = yTrue
      allYPred.append(np.array(yPredProb) if (yPredProb is not None) else np.array(yPred))

    print(f"Total samples across all trials: {len(allYTrue)}")
    print(f"Unique classes: {classes}")

    for which in ["CI", "SD"]:
      fileName = os.path.join(newStorageDir, f"{which}_MultiTrial_PRC_Curve.pdf")
      PlotMultiTrialPRCurve(
        allYTrue,  # List of true labels arrays from all trials.
        allYPred,  # List of predicted probabilities from all trials.
        classes,  # List of class names.
        confidenceLevel=0.95,  # Confidence level for CI.
        which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
        title="Multi-Trial Precision-Recall Curve",
        figSize=(8, 8),  # Figure size in inches.
        cmap=None,  # Colormap for different classes.
        display=False,  # Whether to display the plot.
        save=True,  # Whether to save the plot.
        fileName=fileName,  # File name for saving.
        fontSize=15,  # Font size for labels and annotations.
        showLegend=True,  # Whether to show legend.
        returnFig=False,  # Whether to return the matplotlib figure object.
        dpi=dpi,  # DPI for saving the figure.
        addZoomedInset=True,  # Whether to add a zoomed inset for the top-right corner of the PRC plot.
      )

      fileName = os.path.join(newStorageDir, f"{which}_MultiTrial_ROC_AUC.pdf")
      PlotMultiTrialROCAUC(
        allYTrue,  # List of true labels arrays from all trials.
        allYPred,  # List of predicted probabilities from all trials.
        classes,  # List of class names.
        confidenceLevel=0.95,  # Confidence level for CI (default 95%).
        which=which,  # Method for confidence intervals: "CI" for confidence intervals, "SD" for standard deviation.
        title="Multi-Trial ROC Curve",  # Plot title.
        figSize=(8, 8),  # Figure size.
        cmap=None,  # Colormap for ROC curves.
        display=False,  # Display the plot.
        save=True,  # Save the plot.
        fileName=fileName,  # File name.
        fontSize=15,  # Font size.
        plotDiagonal=True,  # Plot diagonal reference line.
        showLegend=True,  # Show legend.
        returnFig=False,  # Return figure object.
        dpi=dpi,  # DPI for saving the figure.
        addZoomedInset=True,  # Whether to add a zoomed inset for the top-left corner.
      )

    # Save the calculated metrics for each trial to a CSV file for comparison.
    # Example of the file structure (if you have a single system):
    #     Precision, Recall, F1, Accuracy, Specificity, Average
    #     Metric, Metric, Metric, Metric, Metric, Metric
    #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133,
    #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282,
    #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406,
    #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339,
    #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813,
    firstRow = [
      " ".join(el.split(" ")[1:])  # Remove the "Weighted " prefix from the metric names for cleaner column headers.
      for el in kHistory[trialFiles[0]]["metrics"].keys()
      if ("Weighted" in el)
    ]
    secondRow = ["Metric"] * len(firstRow)
    metricValues = [
      [kHistory[trial]["metrics"][f"Weighted {metric}"] for metric in firstRow]
      for trial in trialFiles
    ]
    # Create a DataFrame to store the metrics for each trial, with the first row containing metric names and the
    # second row containing the keyword "Metric".
    dfMetrics = pd.DataFrame(
      data=metricValues,
      columns=firstRow,
    )
    # Insert the secondRow as the second row in the DataFrame at index 0 (after the header); pushes the metric values down by one row.
    dfMetrics.loc[-1] = secondRow  # Add the second row with "Metric" values.
    dfMetrics.index = dfMetrics.index + 1  # Shift the index to accommodate the new row.
    # Sort the index to maintain the correct order (header, "Metric" row, then metric values).
    dfMetrics.sort_index(inplace=True)
    # Save the DataFrame to a CSV file for comparison.
    trialMetricsComparisonFile = os.path.join(newStorageDir, "Trial_Metrics_Comparison.csv")
    dfMetrics.to_csv(trialMetricsComparisonFile, index=False)
    print(f"Trial metrics comparison saved to: {trialMetricsComparisonFile}")
    print(f"Trial Metrics Comparison:\n{dfMetrics}")

    history, names, metrics = ExtractDataFromSummaryFile(trialMetricsComparisonFile)

    newFolderName = ""
    if (plotMetricsIndividual):
      newFolderName = os.path.join(newStorageDir, "PerformanceMetricsPlots")
      os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.

      PlotMetrics(
        history, names, metrics,
        factor=5,  # Factor to multiply the default figure size.
        keyword="AllMetrics",  # Keyword to append to the filenames of the saved plots.
        dpi=dpi,  # Dots per inch (resolution) of the saved plots.
        xTicksRotation=45,  # Rotation angle for x-axis tick labels.
        whichToPlot=[],  # List of plot types to generate.
        fontSize=14,  # Font size for the plots.
        showFigures=False,  # Whether to display the plots or not.
        storeInsideNewFolder=True,  # Whether to store the plots inside a new folder.
        newFolderName=newFolderName,  # Name of the folder to store the plots.
        noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
        cmap="viridis",  # Color map for the plots.
        differentColors=True,  # Whether to use different colors for different plots.
        fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
        fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
        extension=".pdf",  # File extension for saved plots.
      )

    print("\u2713 Performance plots generated.")
    print("\nGenerating statistical analysis report...")
    overallReport = []
    for metric in metrics:
      for index, data in enumerate(history):
        report = StatisticalAnalysis(
          data[metric]["Trials"],
          hypothesizedMean=data[metric]["Mean"],
          secondMetricList=None,
        )
        report["Type"] = names[index]
        report["Metric"] = metric
        overallReport.append(report)
    reportDF = pd.DataFrame(overallReport)
    reportCsvPath = os.path.join(newStorageDir, "Statistical_Analysis_Report.csv")
    reportDF.to_csv(reportCsvPath, index=False)
    print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")

    kHistory.update({
      "allYTrue"                  : allYTrue,
      "allYPred"                  : allYPred,
      "classes"                   : classes,
      "reportDF"                  : reportDF,
      "dfMetrics"                 : dfMetrics,
      "averageMetrics"            : {
        # Calculate the average of the metric values across all trials, skipping the first two rows (header and "Metric" row).
        metric: dfMetrics[metric][1:].mean()
        for metric in firstRow
      },
      "overallReport"             : overallReport,
      "trialMetricsComparisonFile": trialMetricsComparisonFile,
      "statisticsStoragePath"     : newStorageDir,
      "performancePlotsFolder"    : newFolderName,
    })

    allHistory[k] = kHistory

  noOfSystems = len(allHistory.keys())
  print(
    f"\n\u2713 Completed processing all trials for {noOfSystems} systems. "
    f"Aggregated history and results are stored in the `allHistory` dictionary."
  )
  if (noOfSystems > 1):
    print(
      f"\n\u2713 Note: Since multiple top experiments were processed, the `allHistory` dictionary contains "
      f"separate entries for each model with their respective trial results and analyses."
    )

    # Now we need to report the statistics summary across the different top experiments (if there are multiple)
    # for easier comparison.
    for k in allHistory.keys():
      # Example of the file structure (if you have multiple systems):
      #     System A, , , , , , System B, , , , ,
      #     Precision, Recall, F1, Accuracy, Specificity, Average, Precision, Recall, F1, Accuracy, Specificity, Average
      #     0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133, 0.5556, 0.5556, 0.5556, 0.6667, 0.7333, 0.6133
      #     0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282, 0.7692, 0.5556, 0.6452, 0.7708, 0.9000, 0.7282
      #     0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406, 0.8333, 0.2778, 0.4167, 0.7083, 0.9667, 0.6406
      #     0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339, 0.5882, 0.5556, 0.5714, 0.6875, 0.7667, 0.6339
      #     0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813, 0.8000, 0.6667, 0.7273, 0.8125, 0.9000, 0.7813
      systems = list(allHistory.keys())
      metricsNames = allHistory[systems[0]]["dfMetrics"].columns.tolist()
      print("Systems:", systems)
      print("Metrics Names:", metricsNames)

      systemsRow = []
      for system in systems:
        systemsRow.extend([system] + [""] * (len(metricsNames) - 1))
      metricsRow = []
      for system in systems:
        metricsRow.extend(metricsNames)
      dataRows = []
      dfMetrics = allHistory[k]["dfMetrics"]
      # Start from 1 to skip the "Metric" row.
      for i in range(1, len(dfMetrics)):
        row = []
        for system in systems:
          row.extend(dfMetrics.iloc[i].values.tolist())
        dataRows.append(row)
      finalDf = pd.DataFrame(
        dataRows,
        columns=systemsRow,
      )
      # Insert the metricsRowNames as the first row in the DataFrame at index 0 (after the header); pushes the
      # metric values down by one row.
      finalDf.loc[-1] = metricsRow  # Add the first row with system names.
      finalDf.index = finalDf.index + 1  # Shift the index to accommodate the new row.

      # Sort the index to maintain the correct order (header, system names row, then metric values).
      finalDf.sort_index(inplace=True)
      print(finalDf.head())

      # Save the DataFrame to a CSV file for comparison.
      newFolderName = os.path.join(statisticsStoragePath, "Statistics Summary")
      os.makedirs(newFolderName, exist_ok=True)  # Create the folder if it doesn't exist.
      allSystemsMetricsComparisonFile = os.path.join(newFolderName, "All_Systems_Metrics_Comparison.csv")
      finalDf.to_csv(allSystemsMetricsComparisonFile, index=False)
      # Store as Latex table as well for easier inclusion in reports and papers.
      allSystemsMetricsComparisonLatexFile = os.path.join(newFolderName, "All_Systems_Metrics_Comparison.tex")
      with open(allSystemsMetricsComparisonLatexFile, "w") as f:
        f.write(finalDf.to_latex(index=False))
      print(f"All systems metrics comparison saved to: {allSystemsMetricsComparisonFile}")

      # Generate performance metric plots for all systems combined.
      hist, names, metrics = ExtractDataFromSummaryFile(allSystemsMetricsComparisonFile)

      if (plotMetricsOverall):
        PlotMetrics(
          hist, names, metrics,
          factor=5,  # Factor to multiply the default figure size.
          keyword="AllSystems_AllMetrics",  # Keyword to append to the filenames of the saved plots.
          dpi=dpi,  # Dots per inch (resolution) of the saved plots.
          xTicksRotation=45,  # Rotation angle for x-axis tick labels.
          whichToPlot=[],  # List of plot types to generate.
          fontSize=14,  # Font size for the plots.
          showFigures=False,  # Whether to display the plots or not.
          storeInsideNewFolder=True,  # Whether to store the plots inside a new folder.
          newFolderName=newFolderName,  # Name of the folder to store the plots.
          noOfPlotsPerRow=3,  # Number of plots per row in the subplot grid.
          cmap="viridis",  # Color map for the plots.
          differentColors=True,  # Whether to use different colors for different plots.
          fixedTicksColors=True,  # Whether to use fixed ticks colors for consistency across plots.
          fixedTicksColor="black",  # Color to use for fixed ticks if `fixedTicksColors` is True.
          extension=".pdf",  # File extension for saved plots.
        )

      print("\u2713 Performance plots generated.")
      print("\nGenerating statistical analysis report...")
      overallReport = []
      for metric in metrics:
        for index, data in enumerate(hist):
          report = StatisticalAnalysis(
            data[metric]["Trials"],
            hypothesizedMean=data[metric]["Mean"],
            secondMetricList=None,
          )
          report["Type"] = names[index]
          report["Metric"] = metric
          overallReport.append(report)
      reportDF = pd.DataFrame(overallReport)
      reportCsvPath = os.path.join(newFolderName, "All_Systems_Statistical_Analysis_Report.csv")
      reportDF.to_csv(reportCsvPath, index=False)
      # Save the report as Latex table as well for easier inclusion in reports and papers.
      reportLatexPath = os.path.join(newFolderName, "All_Systems_Statistical_Analysis_Report.tex")
      with open(reportLatexPath, "w") as f:
        f.write(reportDF.to_latex(index=False))
      print(f"\u2713 Statistical analysis report saved: {reportCsvPath}")
      print(f"Generated combined performance metric plots for {k} and saved to: {newFolderName}")

  if (noOfSystems > 1):
    print(
      f"\n\u2713 Since multiple top experiments were processed, the `allHistory` dictionary contains "
      f"separate entries for each model with their respective trial results and analyses. "
      f"The statistics summary across the different top experiments has been generated and saved in the "
      f"`Statistics Summary` folder inside the `statisticsStoragePath` directory for easier comparison."
    )
    # Return the top-1 experiment history for each model in the `allHistory` dictionary for further analysis and comparison.
    topSystemDict = {}
    topValue = -1
    for k in allHistory.keys():
      averageMetrics = allHistory[k]["averageMetrics"]
      avg = averageMetrics["Average"]
      if (avg > topValue):
        topValue = avg
        topSystemDict = {
          k: allHistory[k],
        }
    allHistory = topSystemDict
  else:
    print(
      f"\n\u2713 Since there is only one top experiment, the `allHistory` dictionary "
      f"contains the history and results for that single experiment."
    )

  return allHistory


if __name__ == "__main__":
  import numpy as np
  import HMB.MachineLearningHelper as mlh
  import HMB.PerformanceMetrics as pm
  from sklearn.model_selection import train_test_split
  from sklearn.datasets import load_iris

  # Load the Iris dataset.
  data = load_iris()
  X = data.data  # Features.
  y = data.target  # Target labels.

  # Split the dataset into training and testing sets.
  xTrain, xTest, yTrain, yTest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=np.random.randint(0, 10000),
  )

  # Get a scaler object (e.g., Standard Scaler).
  scaler = mlh.GetScalerObject("Standard")

  # Fit the scaler on the training data and transform both training and testing data.
  xTrain = scaler.fit_transform(xTrain)
  xTest = scaler.transform(xTest)
  print(f"xTrain shape: {xTrain.shape}, xTest shape: {xTest.shape}", flush=True)

  # Apply feature selection (e.g., PCA to select 50% of features).
  allFsTechs = ["PCA", "RF", "RFE", "Chi2", "MI", "ANOVA", "LDA"]
  rndFsTech = np.random.choice(allFsTechs)
  xTrain, xTest, fs, features = mlh.PerformFeatureSelection(
    rndFsTech, 50, xTrain, yTrain, xTest, yTest, returnFeatures=True
  )
  print(f"Feature Selection Technique: {rndFsTech}", flush=True)
  print(f"Selected Features: {features}", flush=True)
  print(
    f"xTrain shape after feature selection: {xTrain.shape}, xTest shape after feature selection: {xTest.shape}",
    flush=True
  )

  # Apply data balancing (e.g., SMOTE).
  allBalanceTechs = ["SMOTE", "ADASYN", "BSMOTE", "SVMSMOTE", "ROS", "RUS", "NearMiss", "CCx"]
  rndBalanceTech = np.random.choice(allBalanceTechs)
  xTrain, yTrain, dbObj = mlh.PerformDataBalancing(xTrain, yTrain, techniqueStr=rndBalanceTech)
  print(f"Data Balancing Technique: {rndBalanceTech}", flush=True)
  print(f"xTrain shape after data balancing: {xTrain.shape}", flush=True)
  print(f"yTrain distribution after data balancing: {np.bincount(yTrain)}", flush=True)

  # Apply outlier detection (e.g., Z-Score).
  allOutlierTechs = ["IQR", "ZScore", "IForest", "LOF", "EllipticEnvelope", "OCSVM", "DBSCAN", "Mahalanobis"]
  rndOutlierTech = np.random.choice(allOutlierTechs)
  xTrain, mask = mlh.PerformOutlierDetection(xTrain, techniqueStr=rndOutlierTech, returnMask=True)
  yTrain = yTrain[mask]
  print(f"Outlier Detection Technique: {rndOutlierTech}", flush=True)
  print(f"xTrain shape after outlier detection: {xTrain.shape}", flush=True)
  print(f"yTrain distribution after outlier detection: {np.bincount(yTrain)}", flush=True)
