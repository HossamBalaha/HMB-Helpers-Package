import shap, os, pickle, copy, cv2, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


class SHAPExplainer(object):
  r'''
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

  Example
  -------
  .. code-block:: python

    import HMB.ExplainabilityHelper as eh

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

  Notes
  -----
    SHAP visualizations are saved as PNG and PDF files in the specified storage directory.
    The class supports both global and local interpretability visualizations.
    For more information about SHAP and its visualization techniques, see:
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
    r'''
    Initialize the SHAPExplainer object with file paths and configuration.

    Parameters:
      baseDir (str): Base directory containing data and results.
      experimentFolderName (str): Name of the folder containing model storage files.
      testFilename (str): Filename of the test dataset.
      targetColumn (str): Name of the target column in the dataset.
      pickleFilePath (str): Path to the pickled model/storage file (if not provided, it will be constructed).
      shapStorageKeyword (str): Keyword for the storage path where SHAP results will be saved.
      dpi (int, optional): Dots per inch for saving plots (default: 1080).

    Notes
    -----
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
    r'''
    Load the trained model objects and the test dataset from files, and prepare the test data.

    This method loads the model, scaler, feature selector, and other objects from a pickle file,
    reads the test dataset, applies the same preprocessing pipeline as used during training
    (feature selection, scaling), and limits the number of records if specified.

    Refer to the class "OptunaTuning" documentation in the "MachineLearningHelper" module for details
    on how the model and preprocessing objects are stored.

    Parameters:
      maxNoRecords (int, optional): Maximum number of records to limit the test dataset to (default: 10).

    Notes
    -----
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

    # Apply any feature encoders used during training, if available.
    featuresEncoders = self.objects.get("FeaturesEncoders", None)
    if (featuresEncoders is not None):
      for col, enc in featuresEncoders.items():
        if (col in self.XTest.columns):
          self.XTest[col] = enc.transform(self.XTest[col])

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

    Notes
    -----
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

    Notes
    -----
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

    Notes
    -----
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

    # # Create a list to hold SHAP values for each category.
    # toPlot = [copy.copy(self.shapValues)]
    # for cat in distinctCats:
    #   # Filter SHAP values for the specified category.
    #   shapValuesAlt = copy.copy(self.shapValues)
    #   shapValuesAlt.values = shapValuesAlt.values[self.yTest == cat]  # Filter SHAP values based on the category.
    #   shapValuesAlt.data = shapValuesAlt.data[self.yTest == cat]  # Filter features based on the category.
    #   if (shapValuesAlt.data.shape[0] == 0):
    #     print(f"No records found for category '{cat}'. Skipping this category.")
    #     continue
    #   toPlot.append(shapValuesAlt)

    # Generate SHAP plots for each category (including "All")
    for cat in distinctCats:
      if (cat == "All"):
        temp = copy.copy(self.shapValues)
      else:
        mask = self.yTest == cat
        temp = copy.copy(self.shapValues)
        temp.values = temp.values[mask]
        temp.data = temp.data[mask]
        if (temp.data.shape[0] == 0):
          print(f"No records found for category '{cat}'. Skipping this category.")
          continue

      # # Get the first SHAP values object.
      # temp = toPlot.pop(0)

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


class CAMExplainerPyTorch(object):
  r'''
  A convenience wrapper to run CAM / attribution methods on a torch model and save results.

  This class provides a compact, self-contained interface for computing a wide set of
  class-discriminative and gradient-based attribution maps (Grad-CAM family, Layer-CAM,
  Score-CAM, Ablation-CAM) and classic attribution techniques (saliency, SmoothGrad,
  Integrated Gradients, Occlusion, Grad x Input). The implementation prefers the
  instance-level implementations when available and falls back to module-level helper
  functions present in the same module.

  The class is intended to be used in explainability pipelines where a trained
  PyTorch classification model (or a YOLO classification wrapper) is available and a
  human-readable visualization (heatmap overlay and annotated figure) is required.

  Attributes:
    torchModel (torch.nn.Module | None): The underlying PyTorch model used for inference and gradients.
    yoloModel (object | None): Optional Ultralytics YOLO wrapper from which a torch model may be extracted.
    device (torch.device): Device where model and tensors are executed.
    camType (str): Selected CAM / attribution method name (lowercase key used by dispatch map).
    imgSize (int): Default square input size for preprocessing images.
    alpha (float): Default overlay transparency when blending heatmaps with the image.
    outputBase (Path | None): Optional base path where outputs (Overlays, Heatmaps) are saved.
    figsize (tuple): Default figure size used by annotated visualizations.
    dpi (int): Default DPI used to render annotated images.
    fontSize (int): Base font size used in annotations.
    topN (int): Top-N value used for uncertainty/confidence tracking.
    debug (bool): Enable verbose debug prints if True.
    targetLayer (torch.nn.Module | None): Default convolutional layer chosen as target for CAM computations.

  Example
  -------
  .. code-block:: python

    import torch
    import numpy as np
    from PIL import Image
    from HMB.ExplainabilityHelper import CAMExplainerPyTorch

    # Create a tiny dummy model for a quick smoke test.
    model = torch.nn.Sequential(
      torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
      torch.nn.ReLU(),
      torch.nn.AdaptiveAvgPool2d((8, 8)),
      torch.nn.Flatten(),
      torch.nn.Linear(8 * 8 * 8, 10)
    )

    explainer = CAMExplainerPyTorch(
      torchModel=model,
      device="cpu",
      camType="gradcam",
      imgSize=224,
      outputBase="./ExplainabilityOut",
      debug=True
    )

    # Process a single image and save overlay/annotated outputs.
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    tmpPath = Path("./tmp_sample.png")
    img.save(tmpPath)
    result = explainer.ProcessImage(tmpPath, classNames={i: str(i) for i in range(10)})
    print(result)

  Notes
  -----
    - The class-level implementations are intended to be self-sufficient; if a
      module-level helper function exists with the same name the instance will
      prefer the instance method first and fall back to the module function.
    - Some CAMs (Score-CAM, Ablation-CAM) are computationally heavy for large
      models or high-resolution inputs; tune top-K and sample counts accordingly.
    - Removed methods: RISE, GuidedGradCam, GuidedBackprop, GradientShap and
      DeepLift are intentionally not supported at the class-level and will raise
      a RuntimeError if requested via the class dispatch. Module-level helpers
      (if present) remain unchanged and can be invoked directly.
    - Naming conventions: method names use CamelCase and variables use camelCase.

  '''

  def __init__(
    self,
    torchModel=None,
    yoloModel=None,
    device="cpu",
    camType="gradcam",
    imgSize=640,
    alpha=0.45,
    outputBase=None,
    figsize=(14, 12),
    dpi=300,
    fontSize=14,
    topN=20,
    debug=False,
  ):
    '''
      Initialize the CAMExplainerPyTorch with model, device and visualization settings.

      Parameters:
        torchModel (torch.nn.Module | None): The underlying PyTorch model used for inference and gradients.
        yoloModel (object | None): Optional Ultralytics YOLO wrapper from which a torch model may be extracted.
        device (str): Device where model and tensors are executed ("cpu" or "cuda").
        camType (str): Selected CAM / attribution method name (lowercase key used by dispatch map).
        imgSize (int): Default square input size for preprocessing images.
        alpha (float): Default overlay transparency when blending heatmaps with the image.
        outputBase (Path | None): Optional base path where outputs (Overlays, Heatmaps) are saved.
        figsize (tuple): Default figure size used by annotated visualizations.
        dpi (int): Default DPI used to render annotated images.
        fontSize (int): Base font size used in annotations.
        topN (int): Top-N value used for uncertainty/confidence tracking.
        debug (bool): Enable verbose debug prints if True.

      Notes
      -----
        - If both torchModel and yoloModel are provided, torchModel takes precedence.
        - If no torchModel is provided but a yoloModel is, the underlying torch model is extracted automatically.
    '''

    # Store configuration values.
    self.torchModel = torchModel
    self.yoloModel = yoloModel
    self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    self.camType = camType
    self.imgSize = imgSize
    self.alpha = alpha
    self.outputBase = Path(outputBase) if (outputBase is not None) else None
    # Add cam type subfolder if output base is provided.
    if (self.outputBase is not None):
      self.outputBase = self.outputBase / self.CamTypeToFolderName(self.camType)
      self.outputBase.mkdir(parents=True, exist_ok=True)

    self.figsize = figsize
    self.dpi = dpi
    self.fontSize = fontSize
    self.topN = topN
    self.debug = debug
    # If a YOLO wrapper is provided and no torch model, extract underlying model.
    if ((self.torchModel is None) and (self.yoloModel is not None)):
      try:
        self.torchModel = self.ExtractModel(self.yoloModel)
      except Exception:
        self.torchModel = None
    # Ensure model is on the desired device and set to eval mode.
    if (self.torchModel is not None):
      self.torchModel.to(self.device)
      self.torchModel.eval()
    # Determine a default target convolutional layer for CAM computations.
    self.targetLayer = (
      self.GetLastConvLayer(self.torchModel)
      if (self.torchModel is not None) else None
    )

  def ExtractModel(self, yoloModel):
    r'''
    Extract torch model from a YOLO wrapper or return the same model.

    Parameters:
      yoloModel (object | None): Ultralytics YOLO wrapper or a torch.nn.Module.

    Returns:
      torch.nn.Module | object | None: Extracted underlying torch model when possible, otherwise returns the provided object or None if input is None.

    Notes
    -----
      - This mirrors the module-level helper but lives on the instance so it is
        always available. It tolerates wrappers that nest a `.model` attribute.
    '''

    if (yoloModel is None):
      return None
    try:
      if (hasattr(yoloModel, "model")):
        modelInner = yoloModel.model
        if (hasattr(modelInner, "model")):
          return modelInner.model
        return modelInner
    except Exception:
      pass
    return yoloModel

  def GetLastConvLayer(self, model):
    r'''
    Find the last Conv2d layer to target for Grad-CAM.

    Parameters:
      model (torch.nn.Module | None): PyTorch model to inspect.

    Returns:
      torch.nn.Module | None: The last torch.nn.Conv2d module found in the model or None if no Conv2d layer is present.

    Notes
    -----
      - Traverses the module tree and returns the deepest Conv2d instance. This
        method is safe to call with None and will return None in that case.
    '''

    if (model is None):
      return None
    lastConv = None
    for module in model.modules():
      if (isinstance(module, torch.nn.Conv2d)):
        lastConv = module
    return lastConv

  def NormalizeHeatmap(self, heatmap):
    r'''
    Normalize and enhance heatmap contrast to the [0,1] range.

    Parameters:
      heatmap (np.ndarray): Raw heatmap array with arbitrary range.

    Returns:
      np.ndarray: Normalized and smoothed heatmap clipped to [0,1].

    Notes
    -----
      - Applies clipping, percentile-based contrast stretching, Gaussian blur
        and a mild gamma correction to improve visual contrast.
    '''

    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.size == 0:
      return hm
    hm = np.maximum(hm, 0.0)
    maxVal = hm.max()
    if maxVal <= 1e-8:
      return np.zeros_like(hm)
    hm = hm / maxVal
    p99 = np.percentile(hm, 99.5)
    if p99 > 1e-6:
      hm = np.clip(hm / p99, 0, 1)
    hm = cv2.GaussianBlur(hm, (5, 5), 0)
    hm = np.power(hm, 0.7)
    return np.clip(hm, 0, 1)

  def ApplyHeatmapOverlay(self, imageRgb, heatmap, alpha=None):
    r'''
    Blend heatmap onto an RGB image and return uint8 RGB result.

    Parameters:
      imageRgb (np.ndarray): Original RGB image array (H, W, 3) in uint8 or float.
      heatmap (np.ndarray): Heatmap normalized to [0,1] with shape (H, W).
      alpha (float | None): Blend factor for overlay. If None uses instance alpha.

    Returns:
      np.ndarray: Blended RGB image as uint8.

    Notes
    -----
      - Converts the heatmap to a colormap (Viridis) and blends using cv2.addWeighted.
      - Ensures the heatmap is resized to the image dimensions when needed.
    '''

    if (alpha is None):
      alpha = self.alpha
    heatmapArray = np.asarray(heatmap, dtype=np.float32)
    if heatmapArray.size == 0:
      return np.asarray(imageRgb, dtype=np.uint8)
    heatmapArray = np.clip(heatmapArray, 0, 1)
    hmUint8 = (heatmapArray * 255).astype(np.uint8)
    hmColor = cv2.applyColorMap(hmUint8, cv2.COLORMAP_VIRIDIS)
    hmColor = cv2.cvtColor(hmColor, cv2.COLOR_BGR2RGB)
    base = np.asarray(imageRgb, dtype=np.uint8)
    # Ensure same shape.
    if base.shape[:2] != hmColor.shape[:2]:
      hmColor = cv2.resize(hmColor, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, hmColor, alpha, 0)
    return overlay.astype(np.uint8)

  def LoadImage(self, imagePath, imageSize=None):
    r'''
    Load and preprocess an image for the classifier and return tensor + RGB array.

    Parameters:
      imagePath (Path | str): Path to the image file to load.
      imageSize (int | None): Square size to which the image is resized. If None, uses the explainer instance `imgSize`.

    Returns:
      tuple: (inputTensor, originalImage) where inputTensor is a torch tensor shaped (1, C, H, W) and originalImage is an RGB numpy array (H, W, 3).

    Notes
    -----
      - The pixel intensities are scaled to [0,1] and arranged in CHW order for
        model consumption. The returned originalImage preserves original pixels.
    '''

    if (imageSize is None):
      imageSize = self.imgSize
    image = Image.open(str(imagePath)).convert("RGB")
    imageArray = np.array(image)
    originalImage = imageArray.copy()
    imageResized = cv2.resize(imageArray, (imageSize, imageSize), interpolation=cv2.INTER_LINEAR)
    imageNormalized = imageResized.astype(np.float32) / 255.0
    imageTensor = torch.from_numpy(imageNormalized).permute(2, 0, 1).unsqueeze(0)
    return imageTensor, originalImage

  def CamTypeToFolderName(self, camTypeString):
    r'''
    Return CamelCase folder name for a camType string.

    Parameters:
      camTypeString (str): Lowercase key describing the CAM method.

    Returns:
      str: CamelCase folder name suitable for file system use.

    Notes
    -----
      - Mapping centralizes naming so file outputs use consistent CamelCase
        strings for human-readability.
    '''

    mapping = {
      "gradcam"            : "GradCam",
      "gradcampp"          : "GradCamPP",
      "xgradcam"           : "XGradCam",
      "eigencam"           : "EigenCam",
      "layercam"           : "LayerCam",
      "scorecam"           : "ScoreCam",
      "ablationcam"        : "AblationCam",
      "saliency"           : "Saliency",
      "smoothgrad"         : "SmoothGrad",
      "integratedgradients": "IntegratedGradients",
      "occlusion"          : "Occlusion",
      "gradxinput"         : "GradXInput",
      "smoothgradcampp"    : "SmoothGradCamPP",
    }
    return mapping.get(camTypeString.lower(), camTypeString.title())

  def FormatClassName(self, classIndex, classNames, defaultLabel):
    r'''
    Return readable class name from index.

    Parameters:
      classIndex (int | None): Integer class index to map to a name.
      classNames (dict): Mapping from index to class name.
      defaultLabel (str): Fallback label when no mapping is available.

    Returns:
      str: Resolved class name or the provided defaultLabel.

    Notes
    -----
      - Safe to call with classIndex == None.
    '''

    if (classIndex is None):
      return defaultLabel
    return classNames.get(classIndex, defaultLabel)

  def CreateAnnotatedVisualization(
    self,
    imageRgb,
    heatmap,
    overlayImage,
    className,
    predictedClassName,
    trueClassName,
    alpha,
    confidence,
    methodName="GradCam",
    figureSize=(12, 12),
    dpiValue=300,
    fontSize=14
  ):
    r'''
    Build a 2x2 annotated saliency figure with colorbars.

    Parameters:
      imageRgb (np.ndarray): Original RGB image array.
      heatmap (np.ndarray): Heatmap in [0,1] used to render colorbars.
      overlayImage (np.ndarray): RGB overlay image produced by ApplyHeatmapOverlay.
      className (str): Name of the class being explained.
      predictedClassName (str): Predicted class name for annotation.
      trueClassName (str): Ground truth class name for annotation.
      alpha (float): Transparency value used for the overlay annotation.
      confidence (float): Confidence value for the predicted class in [0,1].
      methodName (str): Human readable method name used in titles.
      figureSize (tuple): Figure size in inches as (W, H).
      dpiValue (int): DPI used when rendering the figure.
      fontSize (int): Base font size used in annotations.

    Returns:
      np.ndarray: RGB numpy array containing the rendered annotated visualization.
    '''

    # Create font size variants for title, panels, and footer.
    fontSizeTitle = int(fontSize * 1.6)
    fontSizePanel = int(fontSize * 1.2)
    fontSizeText = int(fontSize)
    fontSizeFooter = max(10, int(fontSize * 0.9))

    # Build the 2x2 figure grid for original, heatmap (jet), overlay, and heatmap (viridis).
    figure = plt.figure(figsize=(figureSize[0], figureSize[1]), dpi=dpiValue)
    grid = figure.add_gridspec(2, 2, hspace=0, wspace=0.05)

    # Top-left: original image with prediction and optional ground truth.
    axisOriginal = figure.add_subplot(grid[0, 0])
    axisOriginal.imshow(imageRgb)
    axisOriginal.set_title("Original Image", fontsize=fontSizePanel, fontweight="bold", pad=8)
    axisOriginal.axis("off")
    infoText = f"Predicted: {predictedClassName}\nConfidence: {confidence * 100:.1f}%"
    if (trueClassName != "Unknown"):
      infoText += f"\nGround Truth: {trueClassName}"
    axisOriginal.text(
      0.03, 0.95, infoText, transform=axisOriginal.transAxes, fontsize=fontSizeText, va="top",
      bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.9, edgecolor="black", linewidth=1.2)
    )

    # Top-right: heatmap with Jet colormap and colorbar.
    axisHeatmapJet = figure.add_subplot(grid[0, 1])
    imageJet = axisHeatmapJet.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0, interpolation="bilinear")
    axisHeatmapJet.set_title(f"{methodName} (JET)", fontsize=fontSizePanel, fontweight="bold", pad=6)
    axisHeatmapJet.axis("off")
    colorbarJet = plt.colorbar(imageJet, ax=axisHeatmapJet, fraction=0.045, pad=0.03, shrink=0.85)
    colorbarJet.set_label("Importance", rotation=270, labelpad=14, fontsize=fontSizeText, fontweight="bold")
    colorbarJet.ax.tick_params(labelsize=max(10, int(fontSizeText * 0.9)))
    colorbarJet.ax.text(
      1.12, 1.02, "High",
      transform=colorbarJet.ax.transAxes,
      fontsize=max(9, int(fontSizeText * 0.9)),
      color="red",
      fontweight="bold"
    )
    colorbarJet.ax.text(
      1.12, -0.08, "Low",
      transform=colorbarJet.ax.transAxes,
      fontsize=max(9, int(fontSizeText * 0.9)),
      color="blue",
      fontweight="bold"
    )

    # Bottom-left: overlay image with annotation.
    axisOverlay = figure.add_subplot(grid[1, 0])
    axisOverlay.imshow(overlayImage)
    axisOverlay.set_title(rf"Overlay ($\alpha={alpha:.2f}$)", fontsize=fontSizePanel, fontweight="bold", pad=6)
    axisOverlay.axis("off")
    axisOverlay.text(
      0.03, 0.95, f"Explaining predicted: {className}", transform=axisOverlay.transAxes, fontsize=fontSizeText,
      va="top",
      bbox=dict(boxstyle="round,pad=0.6", facecolor="yellow", alpha=0.85, edgecolor="orange", linewidth=1.2)
    )

    # Bottom-right: heatmap with Viridis colormap and colorbar.
    axisHeatmapViridis = figure.add_subplot(grid[1, 1])
    imageViridis = axisHeatmapViridis.imshow(heatmap, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="bilinear")
    axisHeatmapViridis.set_title(f"{methodName} (VIRIDIS)", fontsize=fontSizePanel, fontweight="bold", pad=6)
    axisHeatmapViridis.axis("off")
    colorbarViridis = plt.colorbar(imageViridis, ax=axisHeatmapViridis, fraction=0.045, pad=0.03, shrink=0.85)
    colorbarViridis.set_label("Importance", rotation=270, labelpad=14, fontsize=fontSizeText, fontweight="bold")
    colorbarViridis.ax.tick_params(labelsize=max(10, int(fontSizeText * 0.9)))
    colorbarViridis.ax.text(
      1.12, 1.02, "High", transform=colorbarViridis.ax.transAxes, fontsize=max(9, int(fontSizeText * 0.9)),
      color="yellow", fontweight="bold"
    )
    colorbarViridis.ax.text(
      1.12, -0.08, "Low", transform=colorbarViridis.ax.transAxes, fontsize=max(9, int(fontSizeText * 0.9)),
      color="purple", fontweight="bold"
    )

    # Global title and footer.
    figure.suptitle(f"{methodName} Visualization: {className}", fontsize=fontSizeTitle, fontweight="bold", y=0.97)
    footer = (
      "Maps highlight regions driving the predicted class.\n"
      "Higher colors = stronger evidence. Only predicted class is visualized for clarity."
    )
    figure.text(
      0.5, 0.02, footer, ha="center", fontsize=fontSizeFooter, style="italic",
      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.6, edgecolor="blue", linewidth=1.0)
    )

    # Try to adjust subplots safely and then render to an RGB array.
    try:
      figure.subplots_adjust(left=0.03, right=0.94, top=0.94, bottom=0.02, hspace=0.06, wspace=0.12)
    except Exception:
      pass
    figure.canvas.draw()
    bufferRgba = figure.canvas.buffer_rgba()
    annotatedImage = np.asarray(bufferRgba)[..., :3]
    plt.close(figure)
    return annotatedImage

  def ComputeSaliency(self, inputTensor, predictedClass, targetForCam=None, targetLayer=None):
    r'''
    Dispatch to the requested CAM / attribution routine and return a heatmap.

    Parameters:
      inputTensor (torch.Tensor): Input image tensor shaped (1, C, H, W).
      predictedClass (int): Index of the predicted class returned by the model.
      targetForCam (int | None): Explicit target class index to explain. If None, the predictedClass will be used.
      targetLayer (torch.nn.Module | None): Convolutional layer to use for CAMs.

    Returns:
      np.ndarray: Heatmap normalized to [0,1] as a 2D array matching input spatial dims.

    Notes
    -----
      - Chooses an instance-level implementation when available, otherwise
        falls back to the module-level helper function with the same name.
      - Raises RuntimeError when no implementation is found for the selected camType.
    '''

    targetLayer = targetLayer if (targetLayer is not None) else self.targetLayer
    useTarget = targetForCam if (targetForCam is not None) else predictedClass
    funcMap = {
      "gradcam"            : "ComputeGradCamSaliency",
      "gradcampp"          : "ComputeGradCamPlusPlusSaliency",
      "xgradcam"           : "ComputeXGradCamSaliency",
      "eigencam"           : "ComputeEigenCamSaliency",
      "layercam"           : "ComputeLayerCamSaliency",
      "scorecam"           : "ComputeScoreCamSaliency",
      "ablationcam"        : "ComputeAblationCamSaliency",
      "saliency"           : "ComputeSaliencyMap",
      "smoothgrad"         : "ComputeSmoothGrad",
      "integratedgradients": "ComputeIntegratedGradients",
      "occlusion"          : "ComputeOcclusion",
      "gradxinput"         : "ComputeGradXInput",
      "smoothgradcampp"    : "ComputeSmoothGradCamPlusPlusSaliency",
    }
    chosen = funcMap.get(self.camType, "ComputeGradCamSaliency")
    # If this instance implements a method with that name, call it.
    if (hasattr(self, chosen) and callable(getattr(self, chosen))):
      method = getattr(self, chosen)
      try:
        # Instance methods accept (inputTensor, targetClass, targetLayer=None, device=None).
        return self.NormalizeHeatmap(method(inputTensor, useTarget, targetLayer=targetLayer, device=self.device))
      except TypeError:
        # Try alternate signatures.
        try:
          return self.NormalizeHeatmap(method(inputTensor, useTarget))
        except TypeError:
          pass
    # Fall back to module-level function if present.
    moduleFunc = globals().get(chosen)
    if (moduleFunc is not None):
      try:
        # Some module-level CAM helpers (e.g., Eigen-CAM) do not accept a targetClass
        # argument and instead accept a targetLayer. Call them accordingly.
        if (chosen == "ComputeEigenCamSaliency"):
          return self.NormalizeHeatmap(moduleFunc(self.torchModel, inputTensor, targetLayer, self.device))
        # Default case: functions expect (model, inputTensor, targetClass, targetLayer?, device?).
        try:
          return self.NormalizeHeatmap(moduleFunc(self.torchModel, inputTensor, useTarget, targetLayer, self.device))
        except TypeError:
          return self.NormalizeHeatmap(moduleFunc(self.torchModel, inputTensor, useTarget, device=self.device))
      except Exception as e:
        # Re-raise with context for debugging.
        raise
    # If no implementation found, raise error.
    raise RuntimeError(f"No implementation found for CAM type: {self.camType}")

  def ComputeSmoothGradCamPlusPlusSaliency(
    self,
    inputTensor,
    targetClass,
    targetLayer=None,
    device=None,
    samples=16,
    noiseLevel=0.15
  ):
    r'''
    Compute SmoothGrad-CAM++ by averaging Grad-CAM++ maps over noisy inputs.

    Parameters:
      inputTensor (torch.Tensor): Base input tensor to perturb.
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to attach hooks to for Grad-CAM++.
      device (torch.device | None): Device used for computation.
      samples (int): Number of noisy samples to average.
      noiseLevel (float): Standard deviation of additive gaussian noise.

    Returns:
      np.ndarray: Averaged Grad-CAM++ heatmap normalized to [0,1].

    Notes
    -----
      - This is a smoothing wrapper around the Grad-CAM++ implementation and
        is useful to reduce high-frequency noise in single-shot CAMs.
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for SmoothGrad-CAM++.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    xBase = inputTensor.to(device).detach()
    accumulated = None
    for i in range(samples):
      noise = torch.randn_like(xBase) * noiseLevel
      xNoisy = (xBase + noise).detach()
      try:
        cam = self.ComputeGradCamPlusPlusSaliency(xNoisy, targetClass, targetLayer=targetLayer, device=device)
      except Exception:
        cam = self.ComputeGradCamPlusPlusSaliency(inputTensor, targetClass, targetLayer=targetLayer, device=device)
      camArr = np.asarray(cam, dtype=np.float32)
      if (accumulated is None):
        accumulated = np.zeros_like(camArr, dtype=np.float32)
      accumulated += camArr
    if (accumulated is None):
      return self.ComputeGradCamPlusPlusSaliency(inputTensor, targetClass, targetLayer=targetLayer, device=device)
    avg = accumulated / float(samples)
    avg = avg - avg.min() if avg.size else avg
    if (avg.size and avg.max() > 0):
      avg = avg / float(avg.max())
    return avg.astype(np.float32)

  def ProcessImage(
    self,
    imagePath,
    classNames=None,
    overlaysDir=None,
    heatmapsDir=None,
    contrast=False
  ):
    r'''
    Process a single image: predict, compute saliency and save outputs.

    Parameters:
      imagePath (Path | str): Path to the image file to process.
      classNames (dict | None): Optional mapping class_idx -> className used for annotations.
      overlaysDir (Path | None): Directory to save overlay and annotated PNGs.
      heatmapsDir (Path | None): Directory to save raw heatmap numpy arrays.
      contrast (bool): When True use class-contrast mode (explain top non-predicted class).

    Returns:
      dict: Summary information about the processed image including image path, predicted/true class information and saliency statistics.

    Notes
    -----
      - Prepares output directories when `self.outputBase` was provided at init.
      - File names use CamelCase for the fixed parts to match project conventions.
    '''

    startTime = time.time()

    # Ensure imagePath is a Path object, accepting strings.
    imagePath = Path(imagePath)

    # Load and preprocess the image tensor and obtain the original RGB array.
    inputTensor, originalImage = self.LoadImage(imagePath, imageSize=self.imgSize)

    # Run model forward to obtain logits and probabilities.
    with torch.no_grad():
      output = self.torchModel(inputTensor.to(self.device))
      if (isinstance(output, (list, tuple))):
        output = output[0]
      if (output.dim() == 2):
        logits = output[0]
      elif (output.dim() == 1):
        logits = output
      else:
        raise ValueError(f"Unexpected output shape: {output.shape}")
      predictedClass = int(torch.argmax(logits).item())
      probabilities = torch.softmax(logits, dim=0)
      confidence = float(probabilities[predictedClass].item())
    # Determine target class for CAM when doing class-contrast.
    targetForCam = predictedClass
    if (contrast and (len(probabilities) > 1)):
      probabilitiesNp = probabilities.cpu().numpy()
      sortedIdx = np.argsort(probabilitiesNp)[::-1]
      for alternative in sortedIdx:
        if (alternative != predictedClass):
          targetForCam = int(alternative)
          break
    # Compute the saliency map through the dispatch method.
    saliencyMap = self.ComputeSaliency(
      inputTensor, predictedClass, targetForCam=targetForCam,
      targetLayer=self.targetLayer
    )
    # Resize and overlay the map onto the original image.
    saliencyResized = cv2.resize(
      saliencyMap, (originalImage.shape[1], originalImage.shape[0]),
      interpolation=cv2.INTER_LINEAR
    )
    overlay = self.ApplyHeatmapOverlay(originalImage, saliencyResized, alpha=self.alpha)
    # Resolve class name strings.
    className = self.FormatClassName(predictedClass, classNames or {}, str(predictedClass))
    predictedClassName = className
    parentClass = imagePath.parent.name
    trueClass = None
    try:
      for classIdx, nameVal in (classNames or {}).items():
        if (nameVal == parentClass):
          trueClass = classIdx
          break
    except Exception:
      trueClass = None
    trueClassName = self.FormatClassName(trueClass, classNames or {}, "Unknown")
    # Create annotated visualization using the instance helper.
    annotatedVisualization = self.CreateAnnotatedVisualization(
      originalImage,
      saliencyResized,
      overlay,
      className,
      predictedClassName,
      trueClassName,
      alpha=self.alpha,
      confidence=confidence,
      methodName=self.CamTypeToFolderName(self.camType),
      figureSize=self.figsize,
      dpiValue=self.dpi,
      fontSize=self.fontSize,
    )
    # Prepare output directories and CamelCase filenames.
    if (overlaysDir is None and self.outputBase is not None):
      overlaysDir = self.outputBase / "Overlays"
    if (heatmapsDir is None and self.outputBase is not None):
      heatmapsDir = self.outputBase / "Heatmaps"
    if (overlaysDir is not None):
      overlaysDir.mkdir(parents=True, exist_ok=True)
    if (heatmapsDir is not None):
      heatmapsDir.mkdir(parents=True, exist_ok=True)
    overlayPath = overlaysDir / f"{imagePath.stem}_P{predictedClassName}_C{trueClassName}_Overlay.png"
    annotatedPath = overlaysDir / f"{imagePath.stem}_P{predictedClassName}_C{trueClassName}_Annotated.png"
    heatmapPath = heatmapsDir / f"{imagePath.stem}_P{predictedClassName}_C{trueClassName}_Heatmap.npy"
    # Save outputs to disk.
    Image.fromarray(overlay).save(overlayPath)
    Image.fromarray(annotatedVisualization).save(annotatedPath)
    Image.fromarray(overlay).save(overlayPath.replace(".png", ".pdf"))
    Image.fromarray(annotatedVisualization).save(annotatedPath.replace(".png", ".pdf"))
    np.save(heatmapPath, saliencyResized)
    elapsed = time.time() - startTime
    # Build a summary dictionary to return.
    result = {
      "image"               : str(imagePath),
      "true_class_idx"      : trueClass if (trueClass is not None) else -1,
      "true_class_name"     : trueClassName,
      "predicted_class_idx" : predictedClass,
      "predicted_class_name": predictedClassName,
      "mean_saliency"       : float(np.mean(saliencyResized)),
      "max_saliency"        : float(np.max(saliencyResized)),
      "confidence"          : confidence,
      "processing_time_sec" : elapsed,
      "overlay_path"        : str(overlayPath),
      "annotated_path"      : str(annotatedPath),
      "heatmap_path"        : str(heatmapPath),
      "cam_type"            : self.camType,
    }
    return result

  def ProcessDirectory(self, imageFiles, classNames=None, overlaysDir=None, heatmapsDir=None, contrast=False):
    r'''
    Process a list of images and return results for each image.

    Parameters:
      imageFiles (list[Path] | list[str]): Iterable of image paths to process.
      classNames (dict | None): Optional class index->name mapping for annotations.
      overlaysDir (Path | None): Directory to save overlay/annotated outputs.
      heatmapsDir (Path | None): Directory to save heatmap arrays.
      contrast (bool): When True use class-contrast mode for CAM targets.

    Returns:
      list[dict]: List of result dictionaries returned by `ProcessImage` for each file.

    Notes
    -----
      - Creates output directories if they do not already exist.
    '''

    results = []
    if (self.outputBase is not None):
      if (overlaysDir is None):
        overlaysDir = self.outputBase / "Overlays"
      if (heatmapsDir is None):
        heatmapsDir = self.outputBase / "Heatmaps"
    if (overlaysDir is not None):
      overlaysDir.mkdir(parents=True, exist_ok=True)
    if (heatmapsDir is not None):
      heatmapsDir.mkdir(parents=True, exist_ok=True)
    for idx, imagePath in enumerate(imageFiles, 1):
      try:
        if (self.debug):
          print(f"DEBUG: Processing ({idx}/{len(imageFiles)}): {imagePath}", flush=True)
        result = self.ProcessImage(
          imagePath, classNames=classNames, overlaysDir=overlaysDir, heatmapsDir=heatmapsDir,
          contrast=contrast
        )
        results.append(result)
      except Exception as err:
        print(f"WARNING: Failed to process {imagePath}: {err}", flush=True)
        if (self.debug):
          import traceback
          traceback.print_exc()
    return results

  def ComputeGradCamSaliency(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute Grad-CAM heatmap for the predicted class using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None | int | str): Layer to hook or index/name of layer.
      device (torch.device | None): Device used for computation. If None uses the instance device.

    Returns:
      np.ndarray: Grad-CAM heatmap resized to input spatial dimensions and normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Grad-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    inputData = inputTensor.to(device).detach()
    inputData.requires_grad_(True)

    # Resolve provided targetLayer to a module instance if needed.
    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    activations = []
    gradients = []

    def forwardHook(module, inputValues, outputValues):
      activations.append(outputValues.detach())

    def backwardHook(module, gradientInput, gradientOutput):
      gradients.append(gradientOutput[0].detach())

    forwardHandle = resolvedLayer.register_forward_hook(forwardHook)
    backwardHandle = resolvedLayer.register_full_backward_hook(backwardHook)
    try:
      outputs = model(inputData)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      if (outputs.dim() == 2):
        logits = outputs[0]
      elif (outputs.dim() == 1):
        logits = outputs
      else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")
      score = logits[targetClass]
      model.zero_grad()
      if (inputData.grad is not None):
        inputData.grad.zero_()
      score.backward(retain_graph=True)
      if ((len(activations) == 0) or (len(gradients) == 0)):
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
      activation = activations[-1]
      gradient = gradients[-1]
      weights = gradient.mean(dim=(2, 3), keepdim=True)
      classActivationMap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
      classActivationMap = torch.nn.functional.interpolate(
        classActivationMap, size=(inputData.shape[2], inputData.shape[3]), mode="bilinear", align_corners=False
      )
      cam = classActivationMap.squeeze().cpu().numpy()
      cam = cam - cam.min()
      if (cam.max() > 0):
        cam = cam / cam.max()
      return cam.astype(np.float32)
    finally:
      forwardHandle.remove()
      backwardHandle.remove()

  def ComputeGradCamPlusPlusSaliency(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute Grad-CAM++ heatmap for the predicted class using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to attach hooks to for Grad-CAM++.
      device (torch.device | None): Device used for computation.

    Returns:
      np.ndarray: Grad-CAM++ heatmap normalized to [0,1].

    Notes
    -----
      - This is a smoothing wrapper around the Grad-CAM++ implementation and
        is useful to reduce high-frequency noise in single-shot CAMs.
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Grad-CAM++.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device).detach()
    x.requires_grad_(True)

    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for Grad-CAM++.")

    activations = []
    gradients = []

    def forwardHook(module, inp, out):
      activations.append(out.detach())

    def backwardHook(module, gradIn, gradOut):
      gradients.append(gradOut[0].detach())

    fh = resolvedLayer.register_forward_hook(forwardHook)
    bh = resolvedLayer.register_full_backward_hook(backwardHook)
    try:
      outputs = model(x)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      if (outputs.dim() == 2):
        logits = outputs[0]
      elif (outputs.dim() == 1):
        logits = outputs
      else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")
      score = logits[targetClass]
      model.zero_grad()
      if (x.grad is not None):
        x.grad.zero_()
      score.backward(retain_graph=True)
      if (len(activations) == 0 or len(gradients) == 0):
        raise RuntimeError("Grad-CAM++ hooks did not capture activations/gradients.")
      act = activations[-1]
      grad = gradients[-1]
      grad2 = grad * grad
      grad3 = grad2 * grad
      eps = 1e-8
      alphaNum = grad2
      alphaDen = 2.0 * grad2 + (act * grad3).sum(dim=(2, 3), keepdim=True)
      alpha = alphaNum / (alphaDen + eps)
      weights = (alpha * torch.relu(grad)).sum(dim=(2, 3), keepdim=True)
      cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
      cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
      cam = cam.squeeze().cpu().numpy()
      cam = cam - cam.min()
      if (cam.max() > 0):
        cam = cam / cam.max()
      return cam.astype(np.float32)
    finally:
      fh.remove()
      bh.remove()

  def ComputeXGradCamSaliency(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute XGrad-CAM heatmap for the predicted class using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to attach hooks to for XGrad-CAM.
      device (torch.device | None): Device used for computation.

    Returns:
      np.ndarray: XGrad-CAM heatmap normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for XGrad-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device).detach()
    x.requires_grad_(True)

    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for XGrad-CAM.")

    activations = []
    gradients = []

    def forwardHook(module, inp, out):
      activations.append(out.detach())

    def backwardHook(module, gradIn, gradOut):
      gradients.append(gradOut[0].detach())

    fh = resolvedLayer.register_forward_hook(forwardHook)
    bh = resolvedLayer.register_full_backward_hook(backwardHook)
    try:
      outputs = model(x)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      if (outputs.dim() == 2):
        logits = outputs[0]
      elif (outputs.dim() == 1):
        logits = outputs
      else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")
      score = logits[targetClass]
      model.zero_grad()
      if (x.grad is not None):
        x.grad.zero_()
      score.backward(retain_graph=True)
      if (len(activations) == 0 or len(gradients) == 0):
        raise RuntimeError("XGrad-CAM hooks did not capture activations/gradients.")
      act = activations[-1]
      grad = gradients[-1]
      eps = 1e-8
      weights = (torch.relu(grad) * act).sum(dim=(2, 3), keepdim=True) / (
        grad.abs().sum(dim=(2, 3), keepdim=True) + eps)
      cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
      cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
      cam = cam.squeeze().cpu().numpy()
      cam = cam - cam.min()
      if (cam.max() > 0):
        cam = cam / cam.max()
      return cam.astype(np.float32)
    finally:
      fh.remove()
      bh.remove()

  def ComputeEigenCamSaliency(self, inputTensor, targetLayer=None, device=None):
    r'''
    Compute Eigen-CAM heatmap using activation PCA (gradient-free) using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetLayer (torch.nn.Module | None): Layer to capture activations from.
      device (torch.device | None): Device used for computation.

    Returns:
      np.ndarray: Eigen-CAM heatmap normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Eigen-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    with torch.no_grad():
      x = inputTensor.to(device)

      resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
        self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
      if (resolvedLayer is None):
        raise RuntimeError("No Conv2d layer found for Eigen-CAM.")

      activations = []

      def forwardHook(module, inp, out):
        activations.append(out.detach())

      fh = resolvedLayer.register_forward_hook(forwardHook)
      try:
        outputs = model(x)
        _ = outputs[0] if (isinstance(outputs, (list, tuple))) else outputs
      finally:
        fh.remove()

    if (len(activations) == 0):
      raise RuntimeError("Eigen-CAM hook did not capture activations.")
    act = activations[-1]
    b, c, h, w = act.shape
    actFlat = act.reshape(b, c, h * w)
    cams = []
    for i in range(b):
      a = actFlat[i]
      aCenter = a - a.mean(dim=1, keepdim=True)
      try:
        u, s, v = torch.svd_lowrank(aCenter, q=min(32, min(aCenter.shape) - 1))
        principal = torch.matmul(aCenter.t(), u[:, 0]).reshape(h, w)
      except Exception:
        u, s, v = torch.svd(aCenter)
        principal = torch.matmul(aCenter.t(), u[:, 0]).reshape(h, w)
      principal = torch.relu(principal)
      principal = principal - principal.min()
      if (principal.max() > 0):
        principal = principal / principal.max()
      cams.append(principal.unsqueeze(0))
    cam = torch.stack(cams, dim=0)
    cam = torch.nn.functional.interpolate(
      cam, size=(inputTensor.shape[2], inputTensor.shape[3]), mode="bilinear",
      align_corners=False
    )
    cam = cam.squeeze().cpu().numpy()
    return cam.astype(np.float32)

  def ComputeLayerCamSaliency(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute Layer-CAM heatmap for the predicted class using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to capture activations from.
      device (torch.device | None): Device used for computation.

    Returns:
      np.ndarray: Layer-CAM heatmap normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Layer-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device).detach()
    x.requires_grad_(True)

    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for Layer-CAM.")

    activations = []
    gradients = []

    def forwardHook(module, inp, out):
      activations.append(out.detach())

    def backwardHook(module, gradIn, gradOut):
      gradients.append(gradOut[0].detach())

    fh = resolvedLayer.register_forward_hook(forwardHook)
    bh = resolvedLayer.register_full_backward_hook(backwardHook)
    try:
      outputs = model(x)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      logits = outputs[0] if (outputs.dim() == 2) else outputs
      score = logits[targetClass]
      model.zero_grad()
      if (x.grad is not None):
        x.grad.zero_()
      score.backward(retain_graph=True)
      if (len(activations) == 0 or len(gradients) == 0):
        raise RuntimeError("Layer-CAM hooks did not capture activations/gradients.")
      act = activations[-1]
      grad = gradients[-1]
      layerCam = torch.relu(grad * act).sum(dim=1, keepdim=True)
      layerCam = torch.nn.functional.interpolate(
        layerCam, size=(x.shape[2], x.shape[3]), mode="bilinear",
        align_corners=False
      )
      layerCam = layerCam.squeeze().cpu().numpy()
      layerCam = layerCam - layerCam.min()
      if (layerCam.max() > 0):
        layerCam = layerCam / layerCam.max()
      return layerCam.astype(np.float32)
    finally:
      fh.remove()
      bh.remove()

  def ComputeScoreCamSaliency(self, inputTensor, targetClass, targetLayer=None, device=None, topK=32):
    r'''
    Compute Score-CAM heatmap (forward-based, no gradients) for the predicted class using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to capture channel maps from.
      device (torch.device | None): Device used for computation.
      topK (int): Number of top channels to consider to reduce compute.

    Returns:
      np.ndarray: Score-CAM heatmap normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Score-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device)

    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for Score-CAM.")

    activations = []

    def forwardHook(module, inp, out):
      activations.append(out.detach())

    fh = resolvedLayer.register_forward_hook(forwardHook)
    try:
      with torch.no_grad():
        outputs = model(x)
        if (isinstance(outputs, (list, tuple))):
          outputs = outputs[0]
        logits = outputs[0] if (outputs.dim() == 2) else outputs
      if (len(activations) == 0):
        raise RuntimeError("Score-CAM hook did not capture activations.")
      act = activations[-1]
      b, c, h, w = act.shape
      if (b != 1):
        act = act[:1]
      energy = act.view(c, -1).norm(p=2, dim=1)
      topk = min(topK, c)
      topIdx = torch.topk(energy, k=topk).indices
      weights = []
      for idx in topIdx:
        fmap = act[0, idx]
        fmapUp = torch.nn.functional.interpolate(
          fmap.unsqueeze(0).unsqueeze(0), size=(x.shape[2], x.shape[3]),
          mode="bilinear", align_corners=False
        ).squeeze()
        fmapUp = fmapUp - fmapUp.min()
        if (fmapUp.max() > 0):
          fmapUp = fmapUp / fmapUp.max()
        masked = x * fmapUp.unsqueeze(0)
        with torch.no_grad():
          outMasked = model(masked)
          if (isinstance(outMasked, (list, tuple))):
            outMasked = outMasked[0]
          logitsMasked = outMasked[0] if (outMasked.dim() == 2) else outMasked
          weights.append(logitsMasked[targetClass].item())
      weights = torch.tensor(weights, device=device, dtype=torch.float32)
      weights = torch.relu(weights)
      if (weights.sum() > 0):
        weights = weights / weights.sum()
      cam = torch.zeros((topk, h, w), device=device)
      for i, idx in enumerate(topIdx):
        cam[i] = act[0, idx]
      cam = (weights.view(-1, 1, 1) * cam).sum(dim=0, keepdim=True).unsqueeze(0)
      cam = torch.relu(cam)
      cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
      cam = cam.squeeze().cpu().numpy()
      cam = cam - cam.min()
      if (cam.max() > 0):
        cam = cam / cam.max()
      return cam.astype(np.float32)
    finally:
      fh.remove()

  def ComputeAblationCamSaliency(self, inputTensor, targetClass, targetLayer=None, device=None, topK=32):
    r'''
    Compute Ablation-CAM heatmap by ablating top channels in the target layer using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Layer to capture channel maps from.
      device (torch.device | None): Device used for computation.
      topK (int): Number of top channels to ablate for weight estimation.

    Returns:
      np.ndarray: Ablation-CAM heatmap normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Ablation-CAM.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device)

    resolvedLayer = self.ResolveTargetLayer(model, targetLayer) if (targetLayer is not None) else (
      self.targetLayer if (self.targetLayer is not None) else self.GetLastConvLayer(model))
    if (resolvedLayer is None):
      raise RuntimeError("No Conv2d layer found for Ablation-CAM.")

    activations = []

    def forwardHook(module, inp, out):
      activations.append(out.detach())

    fh = resolvedLayer.register_forward_hook(forwardHook)
    try:
      with torch.no_grad():
        outputs = model(x)
        if (isinstance(outputs, (list, tuple))):
          outputs = outputs[0]
        logitsBase = outputs[0] if (outputs.dim() == 2) else outputs
      if (len(activations) == 0):
        raise RuntimeError("Ablation-CAM hook did not capture activations.")
      act = activations[-1]
      b, c, h, w = act.shape
      if (b != 1):
        act = act[:1]
      energy = act.view(c, -1).norm(p=2, dim=1)
      topk = min(topK, c)
      topIdx = torch.topk(energy, k=topk).indices
      weights = []
      for idx in topIdx:
        mask = torch.ones_like(act)
        mask[:, idx:idx + 1] = 0.0
        maskedAct = act * mask
        handle = resolvedLayer.register_forward_hook(lambda m, i, o: maskedAct)
        try:
          with torch.no_grad():
            outMasked = model(x)
            if (isinstance(outMasked, (list, tuple))):
              outMasked = outMasked[0]
            logitsMasked = outMasked[0] if (outMasked.dim() == 2) else outMasked
            weights.append((logitsBase[targetClass] - logitsMasked[targetClass]).item())
        finally:
          handle.remove()
      weights = torch.tensor(weights, device=device, dtype=torch.float32)
      weights = torch.relu(weights)
      if (weights.sum() > 0):
        weights = weights / weights.sum()
      cam = torch.zeros((topk, h, w), device=device)
      for i, idx in enumerate(topIdx):
        cam[i] = act[0, idx]
      cam = (weights.view(-1, 1, 1) * cam).sum(dim=0, keepdim=True).unsqueeze(0)
      cam = torch.relu(cam)
      cam = torch.nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
      cam = cam.squeeze().cpu().numpy()
      cam = cam - cam.min()
      if (cam.max() > 0):
        cam = cam / cam.max()
      return cam.astype(np.float32)
    finally:
      fh.remove()

  def ResolveTargetLayer(self, model, targetLayer):
    r'''
    Resolve a target layer specification to a torch.nn.Module instance.

    Parameters:
      model (torch.nn.Module): Model containing the target layer.
      targetLayer (torch.nn.Module | int | str | None): Specification of the target layer which can be: (a) None: pick the last Conv2d layer. (b) int: index of the Conv2d layer in model.modules(). (c) str: name of the module in model.named_modules(). (d) torch.nn.Module: already a module instance.

    Returns:
      torch.nn.Module | None: Resolved module instance or None if not found.

    Notes
    -----
      - If targetLayer is None, the last Conv2d layer is selected.
      - If an integer index is provided, the corresponding Conv2d module is selected.
      - If a string name is provided, the named module is searched for.
      - If the targetLayer is already a module-like object with hook API, it is returned as is.
    '''

    # If user passed None, pick the last Conv2d layer using existing helper.
    if (targetLayer is None):
      return self.GetLastConvLayer(model)
    # If an integer index is provided, select the corresponding Conv2d module.
    if (isinstance(targetLayer, int)):
      convs = [m for m in model.modules() if (isinstance(m, torch.nn.Conv2d))]
      if (len(convs) == 0):
        return None
      idx = int(targetLayer)
      if (idx < 0):
        idx = len(convs) + idx
      if (idx < 0 or idx >= len(convs)):
        raise IndexError(f"targetLayer index out of range: {targetLayer}")
      return convs[idx]
    # If a string name is provided, attempt to find a named module.
    if (isinstance(targetLayer, str)):
      for name, mod in model.named_modules():
        if (name == targetLayer):
          return mod
      # fallback to None if not found.
      return None
    # If it's already a module-like object with hook API, return it.
    if (hasattr(targetLayer, "register_forward_hook")):
      return targetLayer
    # Unknown type -> return None.
    return None

  def ComputeIntegratedGradients(self, inputTensor, targetClass, targetLayer=None, device=None, steps=50):
    r'''
    Compute Integrated Gradients for the predicted class from a zero baseline using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Present for API compatibility but not used for Integrated Gradients.
      device (torch.device | None): Device used for computation. If None uses the instance device.
      steps (int): Number of interpolation steps between baseline and input.

    Returns:
      np.ndarray: Integrated Gradients attribution map normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Integrated Gradients.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    baseline = torch.zeros_like(inputTensor).to(device)
    # Build list of scaled inputs excluding the baseline itself.
    scaledInputs = [baseline + (float(k) / steps) * (inputTensor.to(device) - baseline) for k in range(1, steps + 1)]
    totalGrad = np.zeros((inputTensor.shape[2], inputTensor.shape[3]), dtype=np.float32)
    for xScaled in scaledInputs:
      xScaled.requires_grad_(True)
      outputs = model(xScaled)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      logits = outputs[0] if (outputs.dim() == 2) else outputs
      score = logits[targetClass]
      model.zero_grad()
      if (xScaled.grad is not None):
        xScaled.grad.zero_()
      score.backward(retain_graph=False)
      grad = xScaled.grad.detach().cpu().numpy()[0]
      totalGrad += np.mean(grad, axis=0)
    avgGrad = totalGrad / float(steps)
    delta = (inputTensor.detach().cpu().numpy()[0] - baseline.detach().cpu().numpy()[0])
    ig = avgGrad * np.mean(delta, axis=0)
    ig = ig - ig.min()
    if (ig.max() > 0):
      ig = ig / ig.max()
    return ig.astype(np.float32)

  def ComputeOcclusion(self, inputTensor, targetClass, targetLayer=None, device=None, patchSize=32, stride=16):
    r'''
    Compute Occlusion sensitivity map by sliding a gray patch and measuring score drop using the instance model.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Present for API compatibility but not used for Occlusion.
      device (torch.device | None): Device used for computation. If None uses the instance device.
      patchSize (int): Size of square occlusion patch.
      stride (int): Stride to move the occlusion patch.

    Returns:
      np.ndarray: Occlusion sensitivity map normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Occlusion.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)
    x = inputTensor.to(device).detach()
    _, c, H, W = x.shape
    with torch.no_grad():
      baseOutputs = model(x)
      if (isinstance(baseOutputs, (list, tuple))):
        baseOutputs = baseOutputs[0]
      baseLogits = baseOutputs[0] if (baseOutputs.dim() == 2) else baseOutputs
      baseProb = float(torch.softmax(baseLogits, dim=0)[targetClass].item())
    sal = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    for y in range(0, H, stride):
      for x0 in range(0, W, stride):
        y1 = min(y + patchSize, H)
        x1 = min(x0 + patchSize, W)
        xOcc = inputTensor.clone().to(device)
        # Fill occluded region with neutral gray (~0.5 in [0,1]).
        xOcc[:, :, y:y1, x0:x1] = 0.5
        with torch.no_grad():
          outOcc = model(xOcc)
          if (isinstance(outOcc, (list, tuple))):
            outOcc = outOcc[0]
          logitsOcc = outOcc[0] if (outOcc.dim() == 2) else outOcc
          probOcc = float(torch.softmax(logitsOcc, dim=0)[targetClass].item())
        diff = max(0.0, baseProb - probOcc)
        sal[y:y1, x0:x1] += diff
        counts[y:y1, x0:x1] += 1.0
    counts[counts == 0] = 1.0
    sal = sal / counts
    sal = sal - sal.min()
    if (sal.max() > 0):
      sal = sal / sal.max()
    return sal.astype(np.float32)

  def ComputeSaliencyMap(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute vanilla saliency map (absolute gradients) for the target class.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Present for API compatibility but not used for Saliency Map.
      device (torch.device | None): Device used for computation. If None uses the instance device.

    Returns:
      np.ndarray: Saliency map normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for Saliency.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)

    x = inputTensor.to(device).detach()
    x.requires_grad_(True)

    outputs = model(x)
    if (isinstance(outputs, (list, tuple))):
      outputs = outputs[0]
    if (outputs.dim() == 2):
      logits = outputs[0]
    elif (outputs.dim() == 1):
      logits = outputs
    else:
      raise ValueError(f"Unexpected output shape: {outputs.shape}")

    score = logits[targetClass]
    model.zero_grad()
    if (x.grad is not None):
      x.grad.zero_()
    score.backward(retain_graph=False)

    grad = x.grad.detach().cpu().numpy()[0]  # (C, H, W)
    # Aggregate across channels using absolute-mean (robust to sign)
    sal = np.mean(np.abs(grad), axis=0)
    sal = sal - sal.min() if sal.size else sal
    if (sal.size and sal.max() > 0):
      sal = sal / float(sal.max())
    return sal.astype(np.float32)

  def ComputeSmoothGrad(self, inputTensor, targetClass, targetLayer=None, device=None, samples=25, noiseLevel=0.15):
    r'''
    Compute SmoothGrad by averaging saliency maps over noisy input samples.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Present for API compatibility but not used for SmoothGrad.
      device (torch.device | None): Device used for computation. If None uses the instance device.
      samples (int): Number of noisy samples to average over.
      noiseLevel (float): Standard deviation of Gaussian noise relative to input range [0,1].

    Returns:
      np.ndarray: SmoothGrad saliency map normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for SmoothGrad.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)

    xBase = inputTensor.to(device).detach()
    accumulated = None
    for i in range(max(1, int(samples))):
      noise = torch.randn_like(xBase) * float(noiseLevel)
      xNoisy = (xBase + noise).detach()
      xNoisy.requires_grad_(True)

      outputs = model(xNoisy)
      if (isinstance(outputs, (list, tuple))):
        outputs = outputs[0]
      if (outputs.dim() == 2):
        logits = outputs[0]
      elif (outputs.dim() == 1):
        logits = outputs
      else:
        raise ValueError(f"Unexpected output shape: {outputs.shape}")

      score = logits[targetClass]
      model.zero_grad()
      if (xNoisy.grad is not None):
        xNoisy.grad.zero_()
      score.backward(retain_graph=False)

      grad = xNoisy.grad.detach().cpu().numpy()[0]  # (C, H, W)
      sal = np.mean(np.abs(grad), axis=0)
      if (accumulated is None):
        accumulated = np.zeros_like(sal, dtype=np.float32)
      accumulated += sal.astype(np.float32)

    if (accumulated is None):
      return self.ComputeSaliencyMap(inputTensor, targetClass, targetLayer=targetLayer, device=device)

    avg = accumulated / float(max(1, int(samples)))
    avg = avg - avg.min() if avg.size else avg
    if (avg.size and avg.max() > 0):
      avg = avg / float(avg.max())
    return avg.astype(np.float32)

  def ComputeGradXInput(self, inputTensor, targetClass, targetLayer=None, device=None):
    r'''
    Compute gradient * input attributions (Grad x Input) for the target class.

    Parameters:
      inputTensor (torch.Tensor): Input tensor shaped (1, C, H, W).
      targetClass (int): Target class index to explain.
      targetLayer (torch.nn.Module | None): Present for API compatibility but not used for Grad x Input.
      device (torch.device | None): Device used for computation. If None uses the instance device.

    Returns:
      np.ndarray: Grad x Input attribution map normalized to [0,1].
    '''

    model = self.torchModel
    if (model is None):
      raise RuntimeError("No model available on the explainer instance for GradXInput.")
    device = device if (device is not None) else self.device
    model.eval()
    model.to(device)

    x = inputTensor.to(device).detach()
    x.requires_grad_(True)

    outputs = model(x)
    if (isinstance(outputs, (list, tuple))):
      outputs = outputs[0]
    if (outputs.dim() == 2):
      logits = outputs[0]
    elif (outputs.dim() == 1):
      logits = outputs
    else:
      raise ValueError(f"Unexpected output shape: {outputs.shape}")

    score = logits[targetClass]
    model.zero_grad()
    if (x.grad is not None):
      x.grad.zero_()
    score.backward(retain_graph=False)

    grad = x.grad.detach().cpu().numpy()[0]  # (C, H, W)
    inp = inputTensor.detach().cpu().numpy()[0]
    gxi = grad * inp
    sal = np.mean(np.abs(gxi), axis=0)
    sal = sal - sal.min() if sal.size else sal
    if (sal.size and sal.max() > 0):
      sal = sal / float(sal.max())
    return sal.astype(np.float32)
