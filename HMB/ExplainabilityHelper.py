import shap, os, pickle, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SHAPExplainer(object):
  '''
  A class to perform SHAP (SHapley Additive exPlanations) analysis on a trained machine learning model.

  This class provides a pipeline for loading a trained model and its associated data, preparing the test set
  to match the training pipeline (including feature selection and scaling), computing SHAP values for model
  interpretability, and generating a variety of SHAP-based visual explanations (waterfall, force, bar, beeswarm,
  scatter, and summary plots).

  SHAP is a unified approach to explain the output of any machine learning model. It connects game theory with
  local explanations, providing both global and local interpretability.

  Attributes:
    baseDir (str): Base directory containing data and results.
    experimentFolderName (str): Name of the folder containing model storage files.
    testFilename (str): Filename of the test dataset.
    targetColumn (str): Name of the target column in the dataset.
    pickleFilePath (str): Path to the pickled model/storage file.
    shapStorageKeyword (str): Keyword for the storage path where SHAP results will be saved.
    dpi (int): Dots per inch for saving plots.
    storagePath (str): Full path for saving SHAP visualizations.
    objects (dict): Loaded model objects (model, scaler, etc.).
    testData (pd.DataFrame): Loaded test data.
    XTest (pd.DataFrame): Test features.
    yTest (pd.Series): Test target.
    model: Trained model.
    explainer: SHAP explainer object.
    shapValues: Computed SHAP values.
    yPred: Model predictions.
    yPredDecoded: Decoded predictions.

  Example:
    import ExplainabilityHelper as eh
    explainer = eh.SHAPExplainer(
      baseDir="path/to/baseDir",
      experimentFolderName="Experiment1",
      testFilename="test_data.csv",
      targetColumn="target",
      pickleFilePath=None,
      shapStorageKeyword="SHAP_Results"
    )
    explainer.LoadModelAndData(maxNoRecords=100)
    explainer.ComputeShapValues()
    explainer.MakePredictions()
    explainer.VisualizeExplanations(
      instanceIndex=0,
      categoryToExplain="all",
      noOfRecords=150,
      noOfFeatures=5
    )

  Notes:
    - SHAP visualizations are saved as PNG and PDF files in the specified storage directory.
    - The class supports both global and local interpretability visualizations.
    - For more information about SHAP and its visualization techniques, see:
      https://shap.readthedocs.io/en/latest/index.html
  '''

  def __init__(
    self,
    baseDir,
    experimentFolderName,
    testFilename,
    targetColumn,
    pickleFilePath,
    shapStorageKeyword,
    dpi=1080,
  ):
    '''
    Initialize the SHAPExplainer object with file paths and configuration.

    Parameters:
      baseDir (str): Base directory containing data and results.
      experimentFolderName (str): Name of the folder containing model storage files.
      testFilename (str): Filename of the test dataset.
      targetColumn (str): Name of the target column in the dataset.
      pickleFilePath (str): Path to the pickled model/storage file (if not provided, it will be constructed).
      shapStorageKeyword (str): Keyword for the storage path where SHAP results will be saved.
      dpi (int, optional): Dots per inch for saving plots (default: 1080).

    Notes:
      - The storage directory for SHAP results will be created if it does not exist.
      - All attributes are initialized to None except for configuration parameters.
    '''

    self.baseDir = baseDir  # Store the base directory path.
    self.experimentFolderName = experimentFolderName  # Store the storage folder name.
    self.testFilename = testFilename  # Store the test dataset filename.
    self.targetColumn = targetColumn  # Store the target column name.
    self.pickleFilePath = pickleFilePath  # Store the pickle file path.
    self.shapStorageKeyword = shapStorageKeyword  # Store the storage path for results.
    self.dpi = dpi  # Store the DPI for saving plots.

    self.objects = None  # Placeholder for loaded model objects.
    self.testData = None  # Placeholder for loaded test data.
    self.XTest = None  # Placeholder for test features.
    self.yTest = None  # Placeholder for test target.
    self.model = None  # Placeholder for the trained model.
    self.explainer = None  # Placeholder for the SHAP explainer.
    self.shapValues = None  # Placeholder for computed SHAP values.
    self.yPred = None  # Placeholder for model predictions.
    self.yPredDecoded = None  # Placeholder for decoded predictions.

    # Construct the full path for the storage directory.
    self.storagePath = os.path.join(self.baseDir, self.experimentFolderName, self.shapStorageKeyword)
    # Create the storage directory if it does not exist.
    if not os.path.exists(self.storagePath):
      os.makedirs(self.storagePath)

  def LoadModelAndData(self, maxNoRecords=10):
    '''
    Load the trained model objects and the test dataset from files, and prepare the test data.

    This method loads the model, scaler, feature selector, and other objects from a pickle file,
    reads the test dataset, applies the same preprocessing pipeline as used during training
    (feature selection, scaling), and limits the number of records if specified.

    Refer to the class "OptunaTuning" documentation in the "MachineLearningHelper" module for details
    on how the model and preprocessing objects are stored.

    Parameters:
      maxNoRecords (int, optional): Maximum number of records to limit the test dataset to (default: 10).

    Notes:
      - Ensures that the test data columns match those used during training.
      - Applies the same scaler and feature selector as in the training pipeline.
      - If maxNoRecords is set, randomly samples up to that number of records from the test set.
      - Prints the Optuna's best parameters and columns used during training.
    '''

    # Define the path to the file containing the best parameters from Optuna.
    optunaBestParamsFile = os.path.join(
      self.baseDir, self.experimentFolderName, "Optuna Best Params.csv"
    )
    # Load the best parameters from the Optuna file.
    optunaBestParamsDF = pd.read_csv(optunaBestParamsFile)
    # Replace NaN values with "None".
    optunaBestParamsDF.fillna("None", inplace=True)
    # Extract the parameters from the DataFrame.
    optunaBestParams = optunaBestParamsDF.iloc[0].to_dict()

    # Print each parameter and its value.
    print("Optuna Best Parameters:")
    for key, value in optunaBestParams.items():
      print(f"{key}: {value}")

    # Extract model name for potential file naming.
    modelName = optunaBestParams["Model"]

    # Determine the pickle file path to load.
    if (not self.pickleFilePath):
      # Construct pattern if path not directly provided (this logic might need review).
      scalerName = optunaBestParams["Scaler"] if (optunaBestParams["Scaler"] != "None") else None
      fsTech = optunaBestParams["FS Tech"] if (optunaBestParams["FS Tech"] != "None") else None
      fsRatio = optunaBestParams["FS Ratio"] if (optunaBestParams["FS Ratio"] != "None") else None
      dataBalanceTech = optunaBestParams["DB Tech"] if (optunaBestParams["DB Tech"] != "None") else None
      outliersTech = optunaBestParams["Outliers Tech"] if (optunaBestParams["Outliers Tech"] != "None") else None
      pattern = f"{modelName}{scalerName}{fsTech}{fsRatio}{dataBalanceTech}_{outliersTech}.p"
    else:
      pattern = self.pickleFilePath

    # Load the storage dictionary from the pickle file.
    with open(
      os.path.join(self.baseDir, self.experimentFolderName, f"{pattern}"),
      "rb",  # Open the file in read-binary mode.
    ) as f:
      self.objects = pickle.load(f)  # Load the objects (model, scaler, etc.) from the file.

    # Read the test data from the specified CSV file.
    self.testData = pd.read_csv(os.path.join(self.baseDir, self.testFilename))

    # Separate features and target variable from the test data.
    self.XTest = self.testData.drop(columns=[self.targetColumn])  # Drop the target column to get features.
    self.yTest = self.testData[self.targetColumn]  # Extract the target column.

    # Use the columns selected during training to ensure consistency.
    print("Columns used during training:", self.objects["CurrentColumns"])
    self.XTest = self.XTest[self.objects["CurrentColumns"]]
    # Apply the same scaler used during training, if available.
    if (self.objects["Scaler"]):
      self.XTest = self.objects["Scaler"].transform(self.XTest)  # Normalize the features.
      self.XTest = pd.DataFrame(self.XTest, columns=self.objects["CurrentColumns"])  # Convert back to DataFrame.
    # Apply the same feature selector used during training, if available.
    if (self.objects["FeatureSelector"]):
      self.XTest = self.objects["FeatureSelector"].transform(self.XTest)  # Select features.
      self.XTest = pd.DataFrame(self.XTest, columns=self.objects["SelectedFeatures"])  # Convert back to DataFrame.

    # Retrieve the trained model from the loaded objects.
    self.model = self.objects["Model"]

    # Check if the number of records exceeds the maximum limit.
    if (maxNoRecords is not None):
      # Limit the number of records.
      if (self.XTest.shape[0] > maxNoRecords):
        self.XTest = self.XTest.sample(n=maxNoRecords, random_state=42)
        # Ensure target variable matches the sampled features.
        self.yTest = self.yTest.loc[self.XTest.index]

  def ComputeShapValues(self):
    '''
    Initialize the SHAP explainer and compute SHAP values for the test set.

    This method creates a SHAP explainer object using the trained model and the prepared test features,
    then computes SHAP values for the test set to explain model predictions.

    Notes:
      - The computed SHAP values are stored in self.shapValues.
      - Prints the shape of the computed SHAP values.
      - The SHAP explainer is stored in self.explainer.
    '''

    # Initialize SHAP explainer using the trained model and prepared test data.
    self.explainer = shap.Explainer(self.model.predict, self.XTest)

    # Compute SHAP values for the test set to explain model predictions.
    self.shapValues = self.explainer(self.XTest)

    # Display the shape of the computed SHAP values.
    print("SHAP values shape:", self.shapValues.shape)

  def MakePredictions(self):
    '''
    Make predictions on the test set using the loaded model and decode them.

    This method uses the trained model to predict on the prepared test features,
    and decodes the predicted labels back to their original form using the stored label encoder.

    Notes:
      - The predictions are stored in self.yPred.
      - The decoded predictions are stored in self.yPredDecoded.
    '''

    # Make predictions on the prepared test set.
    self.yPred = self.model.predict(self.XTest)
    # Decode the predicted labels back to their original form using the stored label encoder.
    self.yPredDecoded = self.objects["LabelEncoder"].inverse_transform(self.yPred)

  def VisualizeExplanations(
    self,
    instanceIndex=None,
    categoryToExplain="all",
    noOfRecords=150,
    noOfFeatures=5
  ):
    '''
    Generate and save various SHAP visualizations for model interpretability.

    This method produces and saves the following SHAP plots:
      - Waterfall plot for a specific instance's prediction.
      - Force plot for a specific instance's prediction.
      - Bar plot (global feature importance).
      - Beeswarm plot (global feature importance).
      - Scatter and summary plots for the test set, optionally filtered by class/category.

    Parameters:
      instanceIndex (int, optional): Index of the specific instance to explain. If None, a random index is chosen.
      categoryToExplain (int | str, optional): The class label for reference in plots (e.g., 0 for Negative Label = 0). If "all", plots are generated for all classes.
      noOfRecords (int, optional): Number of records to consider for summary/scatter plots.
      noOfFeatures (int, optional): Number of top features to display in plots.

    Notes:
      - All plots are saved as both PNG and PDF files in the storage directory.
      - If categoryToExplain is "all", plots are generated for each unique class in the target variable.
      - Prints a message when visualizations are saved.
      - Uses SHAP's built-in plotting functions for visualization.
    '''
    # Determine the instance index to explain if not provided.
    if (instanceIndex is None):
      instanceIndex = np.random.randint(0, self.XTest.shape[0])  # Choose a random instance index.

    # --- Waterfall Plot ---
    # Visualize the waterfall plot for a specific instance's prediction.
    shap.plots.waterfall(
      self.shapValues[instanceIndex, :noOfFeatures],  # SHAP values for the instance.
      max_display=10,  # Show only the top 10 most important features.
      show=False,  # Prevent automatic display to allow customization.
    )

    # Set the title of the waterfall plot.
    plt.title(
      f"SHAP Waterfall Plot for Instance "
      f"{instanceIndex}\n"
      f"True Label: {self.yTest.iloc[instanceIndex]} and "
      f"Predicted Label: {self.yPredDecoded[instanceIndex]}"
    )
    plt.tight_layout()  # Adjust layout to eliminate wasted space.
    plt.savefig(f"{self.storagePath}/SHAPWaterfallPlot_{instanceIndex}.png", dpi=self.dpi)  # Save the plot as an image.
    plt.savefig(f"{self.storagePath}/SHAPWaterfallPlot_{instanceIndex}.pdf", dpi=self.dpi)  # Save the plot as an image.
    plt.close()  # Close the plot to free up memory.

    # --- Force Plot ---
    # Visualize the force plot for the specific instance's prediction.
    shap.plots.force(
      self.shapValues[instanceIndex, :noOfFeatures],  # SHAP values for the instance.
      matplotlib=True,  # Use Matplotlib for plotting.
      show=False,  # Prevent automatic display.
    )

    # Set the title of the force plot.
    plt.title(
      f"SHAP Force Plot for Instance "
      f"{instanceIndex}\n"
      f"True Label: {self.yTest.iloc[instanceIndex]} and "
      f"Predicted Label: {self.yPredDecoded[instanceIndex]}"
    )
    plt.tight_layout()  # Adjust layout.
    plt.savefig(f"{self.storagePath}/SHAP_Force_Plot_{instanceIndex}.png", dpi=self.dpi)  # Save the plot.
    plt.savefig(f"{self.storagePath}/SHAP_Force_Plot_{instanceIndex}.pdf", dpi=self.dpi)  # Save the plot.
    plt.close()  # Close the plot.

    # --- Bar Plot (Global Feature Importance) ---
    # Visualize the global feature importance using a bar plot.
    shap.plots.bar(
      self.shapValues,  # Pass the full SHAP values to calculate mean absolute values.
      max_display=noOfFeatures,  # Show only the top N most important features.
      show=False,  # Prevent automatic display.
    )

    # Set the title for the global feature importance bar plot.
    plt.title("SHAP Bar Plot (Global Feature Importance)")
    # Adjust layout.
    plt.tight_layout()
    # Save the bar plot as a PNG image.
    plt.savefig(f"{self.storagePath}/SHAP_Bar_Plot_Global.png", dpi=self.dpi)
    # Save the bar plot as a PDF document.
    plt.savefig(f"{self.storagePath}/SHAP_Bar_Plot_Global.pdf", dpi=self.dpi)
    # Close the plot.
    plt.close()

    # --- Beeswarm Plot (Global Feature Importance) ---
    # Visualize the global feature importance using a beeswarm plot.
    shap.plots.beeswarm(
      self.shapValues,  # Pass the full SHAP values.
      max_display=noOfFeatures,  # Show only the top N most important features.
      show=False,  # Prevent automatic display.
    )

    # Set the title for the beeswarm plot.
    plt.title("SHAP Beeswarm Plot (Global Feature Importance)")
    # Adjust layout.
    plt.tight_layout()
    # Save the beeswarm plot as a PNG image.
    plt.savefig(f"{self.storagePath}/SHAP_Beeswarm_Plot_Global.png", dpi=self.dpi)
    # Save the beeswarm plot as a PDF document.
    plt.savefig(f"{self.storagePath}/SHAP_Beeswarm_Plot_Global.pdf", dpi=self.dpi)
    # Close the plot.
    plt.close()

    # --- Scatter and Summary Plots ---
    if (categoryToExplain == "all"):
      # Get all unique categories in the target variable.
      distinctCats = ["All"] + list(self.yTest.unique())
    else:
      distinctCats = [categoryToExplain]

    # Create a list to hold SHAP values for each category.
    toPlot = [copy.copy(self.shapValues)]
    for cat in distinctCats:
      # Filter SHAP values for the specified category.
      shapValuesAlt = copy.copy(self.shapValues)
      shapValuesAlt.values = shapValuesAlt.values[self.yTest == cat]  # Filter SHAP values based on the category.
      shapValuesAlt.data = shapValuesAlt.data[self.yTest == cat]  # Filter features based on the category.
      if (shapValuesAlt.data.shape[0] == 0):
        print(f"No records found for category '{cat}'. Skipping this category.")
        continue
      toPlot.append(shapValuesAlt)

    for cat in distinctCats:
      # Get the first SHAP values object.
      temp = toPlot.pop(0)

      # --- Scatter Plot ---
      # Visualize the scatter plot for the test set.
      shap.plots.scatter(
        temp[:noOfRecords, :noOfFeatures],  # SHAP values for selected records/features.
        show=False,  # Prevent automatic display.
        color=temp[:noOfRecords, :noOfFeatures],  # Color points by their SHAP values.
      )

      # Set the title of the scatter plot.
      plt.title(
        f"SHAP Scatter Plot for the Test Set with "
        f"Reference Class {cat}"
      )
      plt.tight_layout()  # Adjust layout.
      plt.savefig(f"{self.storagePath}/SHAP_Scatter_Plot_{cat}.png", dpi=self.dpi)  # Save the plot.
      plt.savefig(f"{self.storagePath}/SHAP_Scatter_Plot_{cat}.pdf", dpi=self.dpi)  # Save the plot.
      plt.close()  # Close the plot.

      # --- Summary Plot ---
      # Visualize the summary plot for the test set.
      shap.summary_plot(
        temp[:noOfRecords, :noOfFeatures],  # SHAP values for selected records/features.
        show=False,  # Prevent automatic display.
      )

      # Set the title of the summary plot.
      plt.title(
        f"SHAP Summary Plot for the Test Set with "
        f"Reference Class {cat}"
      )
      plt.tight_layout()  # Adjust layout.
      plt.savefig(f"{self.storagePath}/SHAP_Summary_Plot_{cat}.png", dpi=self.dpi)  # Save the plot.
      plt.savefig(f"{self.storagePath}/SHAP_Summary_Plot_{cat}.pdf", dpi=self.dpi)  # Save the plot.
      plt.close()  # Close the plot.

    print(f"SHAP visualizations saved to the {self.storagePath} directory.")
