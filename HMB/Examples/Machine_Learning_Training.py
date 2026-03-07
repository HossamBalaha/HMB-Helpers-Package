import os
from HMB.Initializations import IgnoreWarnings, DoRandomSeeding, UpdateMatplotlibSettings
from HMB.MachineLearningHelper import GetScalerObject, GetMLClassificationModelObject, OptunaTuningClassification
from HMB.PerformanceMetrics import CalculatePerformanceMetrics

if (__name__ == "__main__"):
  IgnoreWarnings()
  DoRandomSeeding()
  UpdateMatplotlibSettings()

  # baseDir = r"path\to\your\directory\which\contains\csv\files"
  baseDir = r"C:\Users\Hossam\Downloads\Features v3"
  csvFiles = [
    el
    for el in os.listdir(baseDir)
    if (el.endswith(".csv"))
  ]
  # Sort the list of CSV files alphabetically.
  csvFiles = sorted(csvFiles)
  print("CSV Files:", csvFiles)

  # TRUE: Perform the training and tuning process.
  # FALSE: Perform the testing and evaluation process using the best parameters obtained from a previous training run.
  IS_TRAINING_PHASE = True

  if (IS_TRAINING_PHASE):
    print("Starting the training and tuning process...")
    for file in csvFiles:
      print(f"Processing file: {file}")
      baseName = os.path.splitext(file)[0]
      tuner = OptunaTuningClassification(
        baseDir=baseDir,
        scalers=["Standard", "MinMax", "MaxAbs", "Robust", "L1", "L2", "Normalizer", "QT", None],
        models=["RF", "LR", "XGB", "SGD", "SVC", "KNN", "AB", "LGBM"],
        fsTechs=["PCA", "RF", "LDA", "RFE", "Chi2", "ANOVA", "MI", None],
        fsRatios=[25, 50, 75, 100] + [10, 20, 30, 40, 60, 70, 80, 90],
        dataBalanceTechniques=[None],
        outliersTechniques=[
          "IQR", "ZScore", "IForest", "LOF", "EllipticEnvelope",
          "OCSVM", "DBSCAN", "Mahalanobis", None
        ],
        datasetFilename=file,
        storageFolderPath=os.path.join(baseDir, f"Optuna_Results_{baseName}"),
        testFilename=None,
        testRatio=0.2,
        contamination=0.05,
        numTrials=250,
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
  else:
    print(
      "Starting the testing and evaluation process using the best parameters "
      "obtained from a previous training run..."
    )
    pass

exit()


# import hyperopt, os, warnings, csv, logging, random, collections, json, tqdm, shap
# # from pyswarm import pso
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import *
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import *
# from sklearn.neighbors import KNeighborsClassifier
# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import *
# from sklearn.metrics import *
# from hyperopt import fmin, tpe, hp, Trials
# from itertools import combinations
# import scipy.stats
# import matplotlib.pyplot as plt
# from feature_selector import FeatureSelector

def MeanCI(data, confidence=0.95):
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
  return h


def LoadDataset(filePath):
  data = pd.read_csv(filePath)
  featuresColumns = data.columns[:-1]
  X = data[featuresColumns]
  y = data.iloc[:, -1].values
  return X, y, featuresColumns


def EncodeLabels(y):
  encoder = LabelEncoder()
  y = encoder.fit_transform(y)
  print("Labels:", encoder.classes_)
  return y


def ClassificationHelper(classifier, X, y, features, cvFolds, scalerName):
  if features is None:
    featuresBool = np.ones(X.shape[1]).astype(bool)
  else:
    featuresBool = np.round(features).astype(bool)
  if featuresBool.sum() == 0:
    return {
      "Accuracy"        : 0,
      "Sensitivity"     : 0,
      "Specificity"     : 0,
      "Precision"       : 0,
      "F1_Score"        : 0,
      "ROC"             : 0,
      "Metrics_Mean"    : 0,
      "Confusion_Matrix": [0, 0, 0, 0],
      "TP"              : 0,
      "TN"              : 0,
      "FP"              : 0,
      "FN"              : 0,
    }
  xSelected = X.values[:, featuresBool]
  xScaled, scaler = GetScalerObject(xSelected, scalerName)
  yPred = cross_val_predict(classifier, xScaled, y, cv=cvFolds, n_jobs=-1, verbose=0)
  metrics = CalculatePerformanceMetrics(y, yPred)
  return metrics


doTrain = False
if (doTrain):
  files = [
    "Features_Fundus_Mar_25_V3_Mod.csv",
    "Features_GLCM_Fundus_Mar_25_V3.csv",
    "Features_GLRLM_Fundus_Mar_25_V3.csv",
    "Features_Hu_Fundus_Mar_25_V3.csv",
    "Features_LBP_Fundus_Mar_25_V3.csv",
    "Features_LTE_Fundus_Mar_25_V3.csv",
    "Features_SFM_Fundus_Mar_25_V3.csv",
    "Features_Shape_Fundus_Mar_25_V3.csv",
    "Features_TAS_Fundus_Mar_25_V3.csv",
  ]

  scalers = [
    None,
    "L1",
    "L2",
    "Max",
    "STD",
    "MinMax",
    "MaxAbs",
    "Robust"
  ]

  classifiers = [
    "CatBoost",
    "DT",
    "SVM",
    "LR",
    "RF",
    "ET",
    "KNN",
    "AdaBoost",
    # "LGBM",
    "XGB",
    # "HGB",
    "MLP",
    "GB",
  ]

  trainSize = 0.85
  historyAll = []

  for datasetPath in files:
    X, y, features = LoadDataset(datasetPath)
    y = EncodeLabels(y)

    # X = X.loc[:, flagsAll]

    xTrain, xTest, yTrain, yTest = train_test_split(
      X, y, train_size=trainSize, random_state=0, stratify=y
    )

    noOfFeatures = X.shape[1]
    noOfSamples = X.shape[0]

    fs = FeatureSelector(data=xTrain, labels=yTrain)
    fs.identify_collinear(correlation_threshold=0.95)
    correlatedFeatures = fs.ops['collinear']
    print(noOfFeatures, len(correlatedFeatures), correlatedFeatures)

    if (len(correlatedFeatures) == 0):
      print(f"File: {datasetPath}, No correlated features found!")
      continue

    history = []

    xTrainSelected = xTrain[correlatedFeatures]
    xTestSelected = xTest[correlatedFeatures]

    for scaler in scalers:
      xTrainScaled, scaler = GetScalerObject(xTrainSelected, scaler)
      xTestScaled = scaler.transform(xTestSelected) if scaler is not None else xTestSelected

      for classifier in classifiers:
        try:
          classifierModel = GetMLClassificationModelObject(classifier, {})
          classifierModel.fit(xTrainScaled, yTrain)

          # yPred = classifierModel.predict(xTestScaled)
          # metricsTest = CalculatePerformanceMetrics(yTest, yPred)

          yPred = classifierModel.predict(X[correlatedFeatures])
          metricsTest = CalculatePerformanceMetrics(y, yPred)

          print(
            f"File: {datasetPath}, Scaler: {scaler}, Classifier: {classifier}, Accuracy: {metricsTest['accuracy']}, Recall: {metricsTest['recall']}"
          )

          history.append(
            {
              "Scaler"    : scaler,
              "Classifier": classifier,
              "File"      : datasetPath,
              **metricsTest,
            }
          )


        except Exception as e:
          print(e)
          continue

    df = pd.DataFrame(history)
    df.to_csv(f"Results_{datasetPath}", index=False)

    historyAll.extend(history)

  df = pd.DataFrame(historyAll)
  df.to_csv(f"Results_Mixed.csv", index=False)

else:
  trainSize = 0.85
  selectedFeatures = [
    'FOS_Median', 'FOS_Entropy', 'FOS_CoefficientOfVariation', 'FOS_75Percentile', 'FOS_90Percentile',
    'FOS_HistogramWidth', 'FOS_Mean_ROI', 'FOS_Variance_ROI', 'FOS_Median_ROI', 'FOS_MaximalGrayLevel_ROI',
    'FOS_75Percentile_ROI', 'FOS_90Percentile_ROI', 'FOS_HistogramWidth_ROI', 'SFM_Roughness_ROI',
    'GLRLM_Short owGrayLevelEmphasis', 'GLRLM_ShortRunHighGrayLevelEmphasis', 'GLRLM_LongRunEmphasis_ROI',
    'GLRLM_HighGrayLevelRunEmphasis_ROI', 'GLRLM_Short owGrayLevelEmphasis_ROI',
    'GLRLM_ShortRunHighGrayLevelEmphasis_ROI', 'GLRLM_LongRunHighGrayLevelEmphasis_ROI', 'GLCM_ASM_Mean',
    'GLCM_Contrast_Mean', 'GLCM_InverseDifferenceMoment_Mean', 'GLCM_SumAverage_Mean', 'GLCM_SumVariance_Mean',
    'GLCM_DifferenceVariance_Mean', 'GLCM_DifferenceEntropy_Mean', 'GLCM_Contrast_Mean_ROI',
    'GLCM_SumOfSquaresVariance_Mean_ROI', 'GLCM_InverseDifferenceMoment_Mean_ROI', 'GLCM_SumAverage_Mean_ROI',
    'GLCM_SumVariance_Mean_ROI', 'GLCM_SumEntropy_Mean_ROI', 'GLCM_DifferenceVariance_Mean_ROI', 'SHAPE_area',
    'SHAPE_YcoordMax_ROI', 'SHAPE_area_ROI', 'SHAPE_perimeter2perArea_ROI', 'LBP_R_1_P_8_entropy',
    'LBP_R_2_P_16_entropy', 'LBP_R_3_P_24_energy', 'LBP_R_3_P_24_entropy', 'LBP_R_4_P_32_energy',
    'LBP_R_4_P_32_entropy', 'LBP_R_5_P_40_energy', 'LBP_R_5_P_40_entropy', 'LBP_R_1_P_8_entropy_ROI',
    'LBP_R_2_P_16_energy_ROI', 'LBP_R_2_P_16_entropy_ROI', 'LBP_R_3_P_24_energy_ROI', 'LBP_R_3_P_24_entropy_ROI',
    'LBP_R_4_P_32_energy_ROI', 'LBP_R_4_P_32_entropy_ROI', 'LBP_R_5_P_40_energy_ROI', 'LBP_R_5_P_40_entropy_ROI',
    'LTE_ES_7_ROI', 'LTE_LS_7_ROI', 'TAS3', 'TAS4', 'TAS5', 'TAS6', 'TAS7', 'TAS9', 'TAS10', 'TAS11', 'TAS12', 'TAS13',
    'TAS14', 'TAS15', 'TAS16', 'TAS17', 'TAS19', 'TAS20', 'TAS21', 'TAS22', 'TAS23', 'TAS24', 'TAS25', 'TAS28', 'TAS29',
    'TAS31', 'TAS32', 'TAS33', 'TAS38', 'TAS40', 'TAS42', 'TAS43', 'TAS46', 'TAS47', 'TAS48', 'TAS49', 'TAS50', 'TAS51',
    'TAS52', 'TAS2_ROI', 'TAS19_ROI', 'TAS20_ROI', 'TAS29_ROI', 'TAS47_ROI'
  ]
  selectedFile = "Features_Fundus_Mar_25_V3_Mod.csv"
  X, yOrig, features = LoadDataset(selectedFile)
  y = EncodeLabels(yOrig)
  X = X[selectedFeatures]
  xTrain, xTest, yTrain, yTest = train_test_split(
    X, y, train_size=trainSize, random_state=0, stratify=y
  )
  xTrainScaled, scaler = GetScalerObject(xTrain, None)
  xTestScaled = scaler.transform(xTest) if None is not None else xTest
  classifierModel = GetMLClassificationModelObject("CatBoost", {})
  classifierModel.fit(xTrainScaled, yTrain)

  samplesIDx = shap.utils.sample(range(len(X)), 500)
  samples = X.iloc[samplesIDx].values
  labels = list(yOrig[samplesIDx])
  print(labels)
  classes = ['GA', 'Intermediate', 'Normal', 'Wet']

  print(samples.shape)
  explainer = shap.Explainer(classifierModel)
  shapValues = explainer.shap_values(samples)
  # shap.summary_plot(shapValues, X100)

  plt.rcParams.update({'font.size': 10})
  shap.decision_plot(
    explainer.expected_value[0],
    shapValues[0],
    X.columns,
    link="logit",
    feature_order="importance",  # or "hclust"
    return_objects=True,
    # highlight=[0, 1, 2, 3],
    # legend_labels=classes,
    # legend_location="lower right",
    auto_size_plot=True,
    show=False,
  )
  plt.rcParams.update({'font.size': 10})
  plt.gcf().set_size_inches(20, 10)
  plt.tight_layout()
  plt.savefig("SHAP_Decision_Plot_500.pdf", dpi=720, bbox_inches='tight')
  plt.savefig("SHAP_Decision_Plot_500.jpg", dpi=720, bbox_inches='tight')

  for idx in [0, 3, 5, 13]:
    shap.force_plot(
      explainer.expected_value[0],
      shapValues[0][idx, :],
      X.iloc[idx, :],
      matplotlib=True,
      show=False,
      figsize=(20, 5),
    )
    plt.rcParams.update({'font.size': 10})
    # plt.gcf().set_size_inches(20, 10)
    plt.tight_layout()
    plt.savefig(f"SHAP_Force_Plot_{labels[idx]}.pdf", dpi=720, bbox_inches='tight')
    plt.savefig(f"SHAP_Force_Plot_{labels[idx]}.jpg", dpi=720, bbox_inches='tight')

# def _ObjectiveFunction(solution):
#   if (solution.sum() == 0):
#     return 0
#
#   solution = (solution > 0).astype(bool)
#   featuresSelected = features[solution.astype(bool)]
#
#   xTrainSelected = xTrain[featuresSelected]
#   xTestSelected = xTest[featuresSelected]
#
#   classifier = GetMLClassificationModelObject("CatBoost", {})
#   classifier.fit(xTrainSelected, yTrain)
#
#   yPred = classifier.predict(xTestSelected)
#   metricsTest = CalculatePerformanceMetrics(yTest, yPred)
#
#   # print(f"Accuracy: {metricsTest['accuracy']}, Recall: {metricsTest['recall']}")
#
#   return metricsTest["Metrics_Mean"]
#
#
# problem = {
#   "obj_func": _ObjectiveFunction,
#   # "bounds"  : BoolVar(n_vars=noOfFeatures, name="delta"),
#   "bounds"  : FloatVar(lb=(-10.0,) * noOfFeatures, ub=(10.0,) * noOfFeatures, name="delta"),
#   "minmax"  : "max",
#   "log_to"  : "console",
# }
#
# ## Run the algorithm
# model = SMA.OriginalSMA(epoch=100, pop_size=50, pr=0.03)
# gBest = model.solve(problem, mode="thread", n_workers=10)
# print(f"Best solution: {gBest.solution}, Best fitness: {gBest.target.fitness}")
# L = list((gBest.solution > 0).astype(bool))
# print(L)


# flagsAll = [
#   True, True, False, False, True, False, True, False, True, True, False, False, False, False, True, False, False, True,
#   False, True, True, False, False, False, True, True, False, True, True, False, False, True, True, True, False, False,
#   False, False, True, False, True, False, False, False, True, False, True, False, False, False, True, True, False,
#   False, False, False, True, True, False, False, True, False, False, False, True, True, False, True, True, True, False,
#   False, True, True, True, True, True, False, False, True, False, True, False, False, False, True, False, True, True,
#   False, True, False, True, False, True, False, True, False, True, True, False, False, True, True, True, False, True,
#   False, True, False, True, False, False, True, True, True, True, True, True, False, False, True, True, True, True,
#   True, True, False, False, True, False, False, False, True, False, False, False, True, True, False, False, False, True,
#   False, False, True, True, False, False, False, True, False, True, False, True, False, True, True, True, True, True,
#   False, True, True, False, False, False, False, False, False, True, True, False, True, True, False, True, True, False,
#   True, False, False, False, False, False, True, False, False, True, False, False, True, False, True, True, True, False,
#   False, False, False, False, True, True, True, True, True, False, False, True, False, True, True, True, True, True,
#   False, True, True, True, True, False, True, False, False, False, False, True, True, True, False, False, False, False,
#   True, True, False, False, False
# ]
