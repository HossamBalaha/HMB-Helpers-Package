# Import TensorFlow for loss implementations.
import tensorflow as tf

# Set the base class to tf.keras.losses.Loss for TensorFlow compatibility.
LossBaseClass = tf.keras.losses.Loss
# Try to get the Reduction enum from tf.keras.losses.
try:
  # Attempt to access the Reduction enum from tf.keras.
  ReductionEnum = tf.keras.losses.Reduction
except AttributeError:
  # Fallback to a simple class with string values if the enum is missing.
  class ReductionEnum:
    # Define SUM_OVER_BATCH_SIZE as mean.
    SUM_OVER_BATCH_SIZE = "mean"
    # Define SUM as sum.
    SUM = "sum"
    # Define NONE as none.
    NONE = "none"


# DiceLoss implements the Dice loss as a Keras Loss subclass.
class DiceLoss(LossBaseClass):
  # Describe the DiceLoss implementation and parameters.
  r'''
  Implements Dice Loss for binary segmentation tasks.
  Dice loss measures the overlap between predicted and ground truth masks.

  .. math::

    \text{Dice} = 1 - \frac{2 \times |X \cap Y| + \text{smooth}}{|X| + |Y| + \text{smooth}}

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.
  '''

  # Initialize the DiceLoss with smoothing constant.
  def __init__(self, smooth=1.0, name="DiceLoss", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to DiceLoss if it is not a string.
      name = "DiceLoss"
    # Store smoothing constant for numerical stability.
    super(DiceLoss, self).__init__(name=name, **kwargs)
    # Store the smoothing constant as a float.
    self.smooth = smooth

  # Call method to compute the Dice loss.
  def call(self, yTrue, yPred):
    # Cast the ground-truth tensor to float32 for numeric stability.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast the prediction tensor to float32 for numeric stability.
    yPred = tf.cast(yPred, tf.float32)
    # Convert logits to probabilities using the sigmoid function.
    probs = tf.nn.sigmoid(yPred)
    # Flatten all elements (batch + spatial) into a single vector.
    probsFlat = tf.reshape(probs, [-1])
    # Flatten the ground-truth tensor into a single vector.
    trueFlat = tf.reshape(yTrue, [-1])
    # Compute the intersection between predictions and ground truth.
    intersection = tf.reduce_sum(probsFlat * trueFlat)
    # Compute the denominator term for Dice computation.
    denom = tf.reduce_sum(probsFlat) + tf.reduce_sum(trueFlat)
    # Compute the Dice score as a global scalar.
    diceScore = (2.0 * intersection + self.smooth) / (denom + self.smooth)
    # Return Dice loss as 1 - Dice score.
    return 1.0 - diceScore


# DiceBCELoss combines binary cross-entropy with Dice loss.
class DiceBCELoss(LossBaseClass):
  # Describe the DiceBCELoss implementation and parameters.
  r'''
  Implements Dice + BCE Loss for binary segmentation tasks.
  Combines Dice loss and binary cross-entropy loss for improved performance on imbalanced data.

  .. math::

    \text{Loss} = \text{BCE}(X, Y) + \left[1 - \frac{2 \times |X \cap Y| + \text{smooth}}{|X| + |Y| + \text{smooth}}\right]

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.

  Note:
    For best practice and autocasting safety, use raw logits as inputs.
  '''

  # Initialize the DiceBCELoss with smoothing constant.
  def __init__(self, smooth=1.0, name="DiceBCELoss", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to DiceBCELoss if it is not a string.
      name = "DiceBCELoss"
    # Store smoothing constant for Dice term and initialize base class.
    super(DiceBCELoss, self).__init__(name=name, **kwargs)
    # Store smoothing constant.
    self.smooth = smooth

  # Call method to compute the combined loss.
  def call(self, yTrue, yPred):
    # Cast ground-truth to float32.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast predictions to float32.
    yPred = tf.cast(yPred, tf.float32)
    # Compute BCE loss from logits using a numerically stable op and average over elements.
    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yTrue, logits=yPred))
    # Convert logits to probabilities for Dice computation.
    probs = tf.nn.sigmoid(yPred)
    # Flatten predictions and ground-truth to 1D vectors.
    probsFlat = tf.reshape(probs, [-1])
    # Flatten ground-truth to 1D vector.
    trueFlat = tf.reshape(yTrue, [-1])
    # Compute intersection and denominator for Dice score.
    intersection = tf.reduce_sum(probsFlat * trueFlat)
    # Compute denominator for Dice score.
    denom = tf.reduce_sum(probsFlat) + tf.reduce_sum(trueFlat)
    # Compute global Dice score and Dice loss.
    diceScore = (2.0 * intersection + self.smooth) / (denom + self.smooth)
    # Compute Dice loss.
    diceLoss = 1.0 - diceScore
    # Return the sum of BCE and Dice losses.
    return bce + diceLoss


# JaccardLoss computes 1 - IoU for binary segmentation.
class JaccardLoss(LossBaseClass):
  # Describe the JaccardLoss implementation and parameters.
  r'''
  Implements Jaccard Loss (IoU Loss) for binary segmentation tasks.
  Jaccard loss measures the intersection over union between predicted and ground truth masks.

  .. math::

    \text{Jaccard} = 1 - \frac{|X \cap Y| + \text{smooth}}{|X \cup Y| + \text{smooth}}

  Parameters:
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.
  '''

  # Initialize the JaccardLoss with smoothing constant.
  def __init__(self, smooth=1.0, name="JaccardLoss", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to JaccardLoss if it is not a string.
      name = "JaccardLoss"
    # Initialize base class and store smoothing constant.
    super(JaccardLoss, self).__init__(name=name, **kwargs)
    # Store smoothing constant.
    self.smooth = smooth

  # Call method to compute the Jaccard loss.
  def call(self, yTrue, yPred):
    # Cast ground-truth to float32.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast predictions to float32.
    yPred = tf.cast(yPred, tf.float32)
    # Convert logits to probabilities.
    probs = tf.nn.sigmoid(yPred)
    # Flatten to 1D vectors for global IoU computation.
    probsFlat = tf.reshape(probs, [-1])
    # Flatten ground-truth to 1D vector.
    trueFlat = tf.reshape(yTrue, [-1])
    # Compute intersection and union components.
    intersection = tf.reduce_sum(probsFlat * trueFlat)
    # Compute total sum of predictions and ground-truth.
    total = tf.reduce_sum(probsFlat + trueFlat)
    # Compute union.
    union = total - intersection
    # Compute Jaccard index and return IoU loss.
    jaccard = (intersection + self.smooth) / (union + self.smooth)
    # Return 1 - Jaccard index.
    return 1.0 - jaccard


# TverskyLoss implements the Tversky index based loss for binary segmentation.
class TverskyLoss(LossBaseClass):
  # Describe the TverskyLoss implementation and parameters.
  r'''
  Implements Tversky Loss for binary segmentation tasks.
  Tversky loss generalizes Dice loss by allowing control over penalties for false positives and false negatives.

  .. math::

    \text{Tversky} = 1 - \frac{|X \cap Y| + \text{smooth}}{|X \cap Y| + \alpha \times |X \setminus Y| + \beta \times |Y \setminus X| + \text{smooth}}

  Parameters:
    alpha (float): Weight for false positives.
    beta (float): Weight for false negatives.
    weight (optional): Not used, for compatibility.
    size_average (optional): Not used, for compatibility.
  '''

  # Initialize the TverskyLoss with alpha, beta and smoothing parameters.
  def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, name="TverskyLoss", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to TverskyLoss if it is not a string.
      name = "TverskyLoss"
    # Initialize base class and store alpha, beta and smoothing parameters.
    super(TverskyLoss, self).__init__(name=name, **kwargs)
    # Store alpha weight safely as a tensor or float.
    self.alpha = tf.cast(alpha, tf.float32) if (not isinstance(alpha, (int, float))) else float(alpha)
    # Store beta weight safely as a tensor or float.
    self.beta = tf.cast(beta, tf.float32) if (not isinstance(beta, (int, float))) else float(beta)
    # Store smoothing constant.
    self.smooth = smooth

  # Call method to compute the Tversky loss.
  def call(self, yTrue, yPred):
    # Cast ground-truth to float32.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast predictions to float32.
    yPred = tf.cast(yPred, tf.float32)
    # Convert logits to probabilities using sigmoid.
    probs = tf.nn.sigmoid(yPred)
    # Flatten predictions and ground-truth to vectors.
    probsFlat = tf.reshape(probs, [-1])
    # Flatten ground-truth to vector.
    trueFlat = tf.reshape(yTrue, [-1])
    # Compute true positives, false negatives and false positives.
    truePos = tf.reduce_sum(probsFlat * trueFlat)
    # Compute false negatives.
    falseNeg = tf.reduce_sum((1.0 - probsFlat) * trueFlat)
    # Compute false positives.
    falsePos = tf.reduce_sum(probsFlat * (1.0 - trueFlat))
    # Compute Tversky index and return loss as 1 - index.
    tverskyIndex = (truePos + self.smooth) / (truePos + self.alpha * falsePos + self.beta * falseNeg + self.smooth)
    # Return 1 - Tversky index.
    return 1.0 - tverskyIndex


# FocalLoss focuses training on hard examples for binary segmentation.
class FocalLoss(LossBaseClass):
  # Describe the FocalLoss implementation and parameters.
  r'''
  Implements Focal Loss for binary segmentation tasks.
  Focal loss focuses training on hard examples and addresses class imbalance.

  .. math::

    \text{Focal}(p_t) = - \alpha \times (1-p_t)^{\gamma} \times \log(p_t)

  Parameters:
    alpha (float): Weighting factor for the rare class. Default is 0.25.
    gamma (float): Focusing parameter for modulating factor (1 - p_t). Default is 2.0.
    reduction (str): Specifies the reduction to apply to the output. Default is "mean".

  Note:
    For autocasting safety, this implementation uses binary_cross_entropy_with_logits directly on logits.
  '''

  # Initialize the FocalLoss with alpha, gamma and reduction parameters.
  def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", name="FocalLoss", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to FocalLoss if it is not a string.
      name = "FocalLoss"
    # Map string reduction to Keras Reduction enum for proper base class handling.
    if (reduction == "mean"):
      # Set reduction enum to SUM_OVER_BATCH_SIZE.
      reductionEnum = ReductionEnum.SUM_OVER_BATCH_SIZE
    elif (reduction == "sum"):
      # Set reduction enum to SUM.
      reductionEnum = ReductionEnum.SUM
    else:
      # Set reduction enum to NONE.
      reductionEnum = ReductionEnum.NONE
    # Initialize base class and store focal loss hyperparameters and reduction method.
    super(FocalLoss, self).__init__(reduction=reductionEnum, name=name, **kwargs)
    # Store alpha parameter.
    self.alpha = alpha
    # Store gamma parameter.
    self.gamma = gamma
    # Store the reduction mode string for internal logic if needed.
    self.reductionMode = reduction

  # Call method to compute the Focal loss.
  def call(self, yTrue, yPred):
    # Cast ground-truth to float32.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast predictions to float32.
    yPred = tf.cast(yPred, tf.float32)
    # Compute per-element binary cross-entropy with logits.
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=yTrue, logits=yPred)
    # Compute probabilities from logits.
    probs = tf.nn.sigmoid(yPred)
    # Flatten arrays to 1D to match PyTorch elementwise operations.
    bceFlat = tf.reshape(bce, [-1])
    # Flatten probabilities to 1D.
    probsFlat = tf.reshape(probs, [-1])
    # Flatten ground-truth to 1D.
    trueFlat = tf.reshape(yTrue, [-1])
    # Compute pt, the probability of the true class for each element.
    pt = tf.where(tf.equal(trueFlat, 1.0), probsFlat, 1.0 - probsFlat)
    # Compute the focal modulating factor for each element.
    focalFactor = self.alpha * tf.pow((1.0 - pt), self.gamma)
    # Compute the focal loss per element.
    loss = focalFactor * bceFlat
    # Return the unreduced loss; the base class will handle reduction automatically.
    return loss


# GeneralizedDiceLoss handles multi-class segmentation with class weighting.
class GeneralizedDiceLoss(LossBaseClass):
  # Describe the GeneralizedDiceLoss implementation and parameters.
  r'''
  Implements Generalized Dice Loss for multi-class segmentation tasks.
  Weights each class inversely to its frequency to address class imbalance.

  .. math::

    \text{Generalized\ Dice} = 1 - \frac{2 \times \sum_c w_c \sum_i p_{ci} \times g_{ci}}{\sum_c w_c \sum_i (p_{ci} + g_{ci})}
    \quad \text{where} \quad w_c = \frac{1}{(\sum_i g_{ci})^2}
  '''

  # Initialize the GeneralizedDiceLoss with epsilon and data format.
  def __init__(self, epsilon=1e-6, name="GeneralizedDiceLoss", dataFormat="channels_last", **kwargs):
    # Ensure name is a string to prevent lambda conversion errors.
    if (not isinstance(name, str)):
      # Set name to GeneralizedDiceLoss if it is not a string.
      name = "GeneralizedDiceLoss"
    # Initialize base class and store epsilon for numerical stability.
    super(GeneralizedDiceLoss, self).__init__(name=name, **kwargs)
    # Store epsilon.
    self.epsilon = epsilon
    # Default data format matches PyTorch (channels_first). Accept "channels_first" or "channels_last".
    self.dataFormat = dataFormat

  # Call method to compute the Generalized Dice loss.
  def call(self, yTrue, yPred):
    # Cast ground-truth to float32 for numeric stability.
    yTrue = tf.cast(yTrue, tf.float32)
    # Cast predictions to float32 for numeric stability.
    yPred = tf.cast(yPred, tf.float32)
    # Apply softmax to logits to obtain class probabilities depending on dataFormat.
    if (self.dataFormat == "channels_first"):
      # Apply softmax over the channel axis for channels-first layout.
      probs = tf.nn.softmax(yPred, axis=1)
      # Determine batch size and class count.
      batchSize = tf.shape(probs)[0]
      # Determine class count.
      classCount = tf.shape(probs)[1]
      # Compute total number of spatial elements.
      spatial = tf.reduce_prod(tf.shape(probs)[2:])
      # Reshape probabilities to (batch, classes, spatial).
      probsFlat = tf.reshape(probs, (batchSize, classCount, spatial))
      # Reshape ground-truth to (batch, classes, spatial).
      trueFlat = tf.reshape(yTrue, (batchSize, classCount, spatial))
    else:
      # Apply softmax over the last axis for channels-last layout.
      probs = tf.nn.softmax(yPred, axis=-1)
      # Determine batch size and class count.
      batchSize = tf.shape(probs)[0]
      # Determine class count.
      classCount = tf.shape(probs)[-1]
      # Compute total number of spatial elements.
      spatial = tf.reduce_prod(tf.shape(probs)[1:-1])
      # Reshape to (batch, spatial, classes).
      tmp = tf.reshape(probs, (batchSize, spatial, classCount))
      # Transpose to (batch, classes, spatial) for consistent downstream ops.
      probsFlat = tf.transpose(tmp, perm=[0, 2, 1])
      # Do the same reshaping and transposing for ground-truth.
      tmpT = tf.reshape(yTrue, (batchSize, spatial, classCount))
      # Transpose ground-truth to (batch, classes, spatial).
      trueFlat = tf.transpose(tmpT, perm=[0, 2, 1])
    # Compute per-class volumes from ground truth.
    vol = tf.reduce_sum(trueFlat, axis=1)
    # Compute class weights as inverse squared volume for balancing.
    w = 1.0 / (tf.square(vol) + self.epsilon)
    # Compute intersection and union per class.
    intersection = tf.reduce_sum(probsFlat * trueFlat, axis=1)
    # Compute union per class.
    union = tf.reduce_sum(probsFlat + trueFlat, axis=1)
    # Weighted numerator per sample.
    numerator = tf.reduce_sum(w * intersection, axis=1)
    # Weighted denominator per sample.
    denominator = tf.reduce_sum(w * union, axis=1)
    # Compute Generalized Dice score and reduce to a scalar loss.
    diceScore = 2.0 * numerator / (denominator + self.epsilon)
    # Return 1 - mean of Dice score.
    return 1.0 - tf.reduce_mean(diceScore)


# Execute the main block if this script is run directly.
if (__name__ == "__main__"):
  # Example usage for all loss functions.
  # Simulate model output tensor with random values (logits).
  logits = tf.random.normal((1, 256, 256, 1))
  # Simulate binary targets.
  binaryTargets = tf.cast(tf.random.uniform((1, 256, 256, 1)) > 0.5, tf.float32)
  # Instantiate loss objects.
  dLoss = DiceLoss()
  # Instantiate DiceBCELoss object.
  dbLoss = DiceBCELoss()
  # Instantiate JaccardLoss object.
  jLoss = JaccardLoss()
  # Instantiate TverskyLoss object.
  tLoss = TverskyLoss(alpha=0.7, beta=0.3)
  # Instantiate FocalLoss object.
  fLoss = FocalLoss()
  # Compute and print Dice loss.
  print("Dice Loss:", float(dLoss(binaryTargets, logits)))
  # Compute and print Dice+BCE loss.
  print("Dice+BCE Loss:", float(dbLoss(binaryTargets, logits)))
  # Compute and print Jaccard loss.
  print("Jaccard Loss:", float(jLoss(binaryTargets, logits)))
  # Compute and print Tversky loss.
  print("Tversky Loss:", float(tLoss(binaryTargets, logits)))
  # Compute and print Focal loss.
  print("Focal Loss:", float(fLoss(binaryTargets, logits)))
  # Multi-class example for Generalized Dice.
  mcLogits = tf.random.normal((2, 128, 128, 3))
  # Simulate multi-class targets.
  mcTargets = tf.one_hot(tf.random.uniform((2, 128, 128), maxval=3, dtype=tf.int32), depth=3)
  # Instantiate GeneralizedDiceLoss object.
  gdl = GeneralizedDiceLoss()
  # Instantiate with channels_last for this NHWC test.
  gdl = GeneralizedDiceLoss(dataFormat="channels_last")
  # Compute and print Generalized Dice loss.
  print("Generalized Dice Loss:", float(gdl(mcTargets, mcLogits)))
