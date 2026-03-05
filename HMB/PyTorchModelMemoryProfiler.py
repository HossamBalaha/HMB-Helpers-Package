import json, time, torch
import torch.nn as nn
from typing import Dict, Tuple, Any, List, Optional


# Define a profiler class for measuring model memory and compute characteristics.
class PyTorchModelMemoryProfiler:
  r'''
  Profiler for estimating PyTorch model memory usage (parameters, gradients,
  activations, optimizer state, attention matrices) and compute (FLOPs/GFLOPs)
  for training and inference. Uses a dummy forward pass with hooks to gather
  activation shapes and sizes.

  Parameters:
    model (nn.Module):  The PyTorch model to profile.
    inputShape (Tuple[int, ...]):  Shape of a single input sample (channels, H, W, ...).
    batchSize (int):  Batch size used for the dummy forward pass. Defaults to 1.
    precision (str):  "FP32" or "FP16" -- affects bytes-per-parameter calculations.
    device (str):  Target device string (e.g., "cpu" or "cuda"). Used for the
      dummy forward pass attempt; profiling logic falls back to CPU if unavailable.
  '''

  # Initialize the profiler with model and input configuration.
  def __init__(
      self, model: nn.Module,
      inputShape: Tuple[int, ...],
      batchSize: int = 1,
      precision: str = "FP32",
      device: str = "cpu"
  ):
    # Store the model reference for later analysis.
    self.model = model

    # Store the input shape for dummy input creation.
    self.inputShape = inputShape

    # Store the batch size for memory calculations.
    self.batchSize = batchSize

    # Validate the precision selection and raise if invalid.
    if (precision not in ["FP32", "FP16"]):
      raise ValueError("Precision must be either 'FP32' or 'FP16'.")

    # Store the precision selection.
    self.precision = precision

    # Compute bytes per parameter based on precision selection.
    self.bytesPerParam = 4 if (self.precision == "FP32") else 2

    # Store the target device for the dummy forward pass.
    self.device = device

    # Initialize list for activation memory entries.
    self.activationMemoryList: List[Dict[str, Any]] = []

    # Initialize list for layer-level information.
    self.layerInfoList: List[Dict[str, Any]] = []

    # Initialize storage for hook handles to allow cleanup.
    self._hookHandles: List[torch.utils.hooks.RemovableHandle] = []

  # Register forward hooks on leaf modules to capture activation metadata.
  def _RegisterForwardHooks(self) -> None:
    r'''
    Attach forward hooks to leaf (no-children) modules in the model so that
    their output activations (shapes and bytes) are recorded during a
    dummy forward pass.
    '''

    # Remove any previously registered hooks to avoid duplicates.
    for handle in getattr(self, "_hookHandles", []):
      try:
        handle.remove()
      except Exception:
        pass
    # Reset the hook handles list after removal.
    self._hookHandles = []

    # Build a mapping of module names to module objects.
    namedModules = dict(self.model.named_modules())

    # Initialize a counter to track repeated occurrences of the same module.
    occurrenceCounter: Dict[str, int] = {}

    # Define a factory that returns a forward hook capturing the provided name.
    def MakeHook(name: str):
      r'''
      Factory that creates a forward hook function bound to a module name.

      Parameters:
        name (str):  Readable name for the module used in recorded entries.

      Returns:
        function:  A hook function suitable for register_forward_hook.
      '''

      # The hook function will record activation shapes and sizes.
      def HookFunction(module: nn.Module, input: Tuple[torch.Tensor, ...], output) -> None:
        r'''
        Forward hook that records output tensor shapes, element counts, and
        per-module parameter counts into self.activationMemoryList when the
        module produces tensor outputs.

        Parameters:
          module (nn.Module):  The module being executed.
          input (Tuple[torch.Tensor, ...]):  The inputs passed to the module.
          output:  The output produced by the module (tensor, tuple, or list).
        '''

        # Initialize a list to collect tensor outputs from the module.
        outputs: List[torch.Tensor] = []

        # If the output is a tensor, record it.
        if (isinstance(output, torch.Tensor)):
          outputs = [output]

        # If the output is a list or tuple, collect tensor elements.
        elif (isinstance(output, (list, tuple))):
          for o in output:
            if (isinstance(o, torch.Tensor)):
              outputs.append(o)

        # If there are no tensor outputs, return early.
        if (len(outputs) == 0):
          return

        # Compute total number of elements across all outputs.
        activationNumel = sum(o.numel() for o in outputs)

        # Compute activation memory in bytes using bytesPerParam.
        # Use actual dtype of the first output tensor for accuracy.
        firstOutput = outputs[0]
        activationBytes = activationNumel * firstOutput.element_size()

        # Build a list of input tensor shapes for the module.
        inputShapes: List[Tuple[int, ...]] = []
        for inp in input:
          if (isinstance(inp, torch.Tensor)):
            inputShapes.append(tuple(inp.shape))

        # Update occurrence counter for the module name.
        idx = occurrenceCounter.get(name, 0) + 1
        occurrenceCounter[name] = idx

        # Compute parameter count local to this module (non-recursive).
        moduleParamCount = 0
        for p in module.parameters(recurse=False):
          moduleParamCount += p.numel()

        # Append a structured activation entry into the activationMemoryList.
        self.activationMemoryList.append({
          "ModuleName"           : name,
          "LayerType"            : module.__class__.__name__,
          "ActivationMemoryBytes": activationBytes,
          "OutputShape"          : tuple(outputs[0].shape) if (len(outputs) > 0) else None,
          "AllOutputShapes"      : [tuple(o.shape) for o in outputs],
          "InputShapes"          : inputShapes,
          "OccurrenceIndex"      : idx,
          "ModuleParamCount"     : moduleParamCount
        })

      # Return the constructed hook function.
      return HookFunction

    # Iterate named modules and attach hooks to leaf modules only.
    for name, module in namedModules.items():
      # Choose a readable hook name for the root if necessary.
      hookName = name if (name != "") else module.__class__.__name__

      # Attach the hook only to leaf modules that have no children.
      if (len(list(module.children())) == 0):
        handle = module.register_forward_hook(MakeHook(hookName))
        self._hookHandles.append(handle)

  # Count total, trainable, and non-trainable parameters in the model.
  def _CountParameters(self) -> Dict[str, int]:
    r'''
    Walk model parameters and count total, trainable (requires_grad) and
    non-trainable parameter elements.

    Returns:
      Dict[str, int]:  Dictionary with keys "TotalParameters", "TrainableParameters", and "NonTrainableParameters".
    '''

    # Initialize counters for total and trainable parameters.
    totalParams = 0
    trainableParams = 0

    # Iterate over all model parameters to accumulate counts.
    for param in self.model.parameters():
      # Compute number of elements in the current parameter tensor.
      paramCount = param.numel()

      # Accumulate into the total parameter counter.
      totalParams += paramCount

      # If the parameter requires gradient, count it as trainable.
      if (param.requires_grad):
        trainableParams += paramCount

    # Compute non-trainable parameters by subtraction.
    nonTrainableParams = totalParams - trainableParams

    # Return structured counts as a dictionary.
    return {
      "TotalParameters"       : totalParams,
      "TrainableParameters"   : trainableParams,
      "NonTrainableParameters": nonTrainableParams
    }

  # Count buffer tensors (e.g., running stats) and their memory usage.
  def _CountBuffers(self) -> Dict[str, int]:
    r'''
    Count registered buffers (such as running_mean/running_var in BatchNorm)
    and estimate their memory usage in bytes using each buffer's element size.

    Returns:
      Dict[str, int]:  Dictionary with keys "TotalBufferElements" and "BufferMemoryBytes".
    '''

    # Initialize counters for buffer elements and bytes.
    totalBuffers = 0
    bufferBytes = 0

    # Iterate over model buffers and accumulate element counts.
    for buf in self.model.buffers():
      totalBuffers += buf.numel()
      # Use actual element size for buffer byte calculation.
      bufferBytes += buf.numel() * buf.element_size()

    # Return buffer statistics.
    return {
      "TotalBufferElements": totalBuffers,
      "BufferMemoryBytes"  : bufferBytes
    }

  # Estimate memory consumed by attention matrices for transformer models.
  def _EstimateAttentionMemory(self, sequenceLength: int, numHeads: int = 8, numLayers: int = 12) -> int:
    r'''
    Estimate memory used by attention score matrices for a transformer-style
    model, which scale quadratically with sequence length.

    Parameters:
      sequenceLength (int):  Sequence length (N) used in attention.
      numHeads (int):  Number of attention heads. Defaults to 8.
      numLayers (int):  Number of layers with attention. Defaults to 12.

    Returns:
      int:  Estimated total attention memory in bytes.
    '''

    # Compute the number of elements in a single attention matrix per layer and head.
    attentionMatrixSize = self.batchSize * numHeads * sequenceLength * sequenceLength

    # Convert element count to bytes and multiply by number of layers.
    totalAttentionMemory = attentionMatrixSize * numLayers * self.bytesPerParam

    # Return the estimated attention memory in bytes.
    return totalAttentionMemory

  # Estimate memory used by optimizer state based on optimizer type and options.
  def _EstimateOptimizerStateMemory(
      self,
      trainableParams: int,
      optimizerType: str = "Adam",
      optimizerKwargs: Optional[Dict[str, Any]] = None
  ) -> int:
    r'''
    Estimate memory required for optimizer state tensors (e.g., Adam's m and
    v buffers) given the number of trainable parameters and optimizer type.

    Parameters:
      trainableParams (int):  Number of trainable parameter elements.
      optimizerType (str):  Optimizer name (e.g., "Adam", "SGD"). Defaults to "Adam".
      optimizerKwargs (Optional[Dict[str, Any]]):  Optimizer options used to
        determine additional state requirements (e.g., amsgrad, momentum).

    Returns:
      int:  Estimated optimizer state memory in bytes for the model.
    '''

    # Normalize kwargs for safe access.
    opts = optimizerKwargs or {}

    # Decide number of state variables per parameter depending on optimizer type and options.
    if (optimizerType in ["Adam", "AdamW"]):
      # Adam / AdamW normally keep two state tensors per param (m and v).
      stateVariablesPerParam = 2

      # If AMSGrad variant requested, add a third buffer.
      if (opts.get("amsgrad", False)):
        stateVariablesPerParam = 3

    # Handle SGD with optional momentum.
    elif (optimizerType == "SGD"):
      # SGD without momentum has no extra state; with momentum it has one momentum buffer.
      if (opts.get("momentum", 0) > 0):
        stateVariablesPerParam = 1
      else:
        stateVariablesPerParam = 0

    # Handle Adagrad which maintains one accumulator per parameter.
    elif (optimizerType == "Adagrad"):
      stateVariablesPerParam = 1

    # Handle RMSprop which has one or two state variables depending on centering.
    elif (optimizerType == "RMSprop"):
      if (opts.get("centered", False)):
        stateVariablesPerParam = 2
      else:
        stateVariablesPerParam = 1

    # Default conservative fallback for unknown optimizers.
    else:
      stateVariablesPerParam = 1

    # Compute optimizer state memory assuming same dtype as model parameters.
    optimizerMemory = trainableParams * stateVariablesPerParam * self.bytesPerParam

    # Return optimizer state memory estimate.
    return optimizerMemory

  # Estimate FLOPs for common layers using recorded activation shapes.
  def _EstimateFLOPs(self) -> Dict[str, Any]:
    r'''
    Estimate layer-wise and total FLOPs using recorded activation shapes
    collected during the dummy forward pass. Supports common layers like
    Conv2d, Linear, BatchNorm, pooling, activations and MultiheadAttention.

    Returns:
      Dict[str, Any]:  Dictionary containing "TotalFLOPs", "PerLayerFLOPs", and "TotalGFLOPs" (rounded conversion to GFLOPs).
    '''

    # Initialize running totals for FLOPs and per-layer details.
    totalFlops = 0
    perLayerFlops: List[Dict[str, Any]] = []

    # Create a mapping from module names to module instances.
    nameToModule = dict(self.model.named_modules())

    # Iterate recorded activation entries to estimate per-layer FLOPs.
    for entry in self.activationMemoryList:
      # Extract stored fields from the activation entry.
      name = entry.get("ModuleName")
      layerType = entry.get("LayerType")
      outShape = entry.get("OutputShape")
      occ = entry.get("OccurrenceIndex", 1)

      # Attempt to find the corresponding module object.
      module = nameToModule.get(name, None)

      # Initialize flops estimate for this layer.
      flops = 0

      # Compute flops for known layer types inside a safe try/except.
      try:
        # Skip if module is not available.
        if (module is None):
          flops = 0

        # Compute Conv2d FLOPs using weight shape and output spatial dimensions.
        elif (isinstance(module, nn.Conv2d)):
          # Read weight tensor and shape details for convolution.
          weight = module.weight
          Cout, CinPerGroup, kH, kW = weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]
          groups = module.groups if hasattr(module, "groups") else 1
          if (outShape is not None and len(outShape) >= 3):
            Hout, Wout = outShape[-2], outShape[-1]
            batch = outShape[0] if (len(outShape) == 4) else self.batchSize
            # Use CinPerGroup directly; weight.shape[1] already equals in_channels/groups.
            flops = 2 * Cout * Hout * Wout * CinPerGroup * kH * kW
            flops *= batch

        # Compute Linear FLOPs assuming a dense matrix multiply per batch.
        elif (isinstance(module, nn.Linear)):
          # Read weight shape for linear layer.
          weight = module.weight
          outF, inF = weight.shape[0], weight.shape[1]
          batch = outShape[0] if (outShape is not None and len(outShape) >= 2) else self.batchSize
          flops = 2 * inF * outF * batch

        # Compute BatchNorm FLOPs as affine transform per element if affine parameters exist.
        elif (isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))):
          if (outShape is not None):
            # Estimate as 2 FLOPs per output element for scale and shift.
            numElems = 1
            for d in outShape:
              numElems *= d
            flops = 2 * numElems

        # Compute pooling FLOPs for MaxPool and AvgPool using kernel arithmetic cost.
        elif (isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))):
          if (outShape is not None):
            batch = outShape[0]
            channels = outShape[1] if (len(outShape) >= 3) else 1
            Hout = outShape[-2]
            Wout = outShape[-1]
            k = module.kernel_size
            # Normalize kernel size tuple.
            if (isinstance(k, int)):
              kH = k
              kW = k
            else:
              kH, kW = k
            # AvgPool does adds/divides; MaxPool does comparisons.
            # Approximate cost as kH*kW ops per output element.
            flops = batch * channels * Hout * Wout * (kH * kW)

        # Compute elementwise operations like ReLU, Sigmoid, and Tanh based on activation size.
        elif (isinstance(module, nn.ReLU)):
          if (outShape is not None):
            numElems = 1
            for d in outShape:
              numElems *= d
            # ReLU is a single comparison per element.
            flops = numElems

        elif (isinstance(module, (nn.Sigmoid, nn.Tanh))):
          if (outShape is not None):
            numElems = 1
            for d in outShape:
              numElems *= d
            # Sigmoid/Tanh are more expensive; approximate as 4 FLOPs per element.
            flops = 4 * numElems

        # Compute a very rough MultiheadAttention FLOPs estimate.
        elif (isinstance(module, nn.MultiheadAttention)):
          if (outShape is not None and len(outShape) >= 2):
            batch = outShape[0]
            seq = outShape[1]
            embed = getattr(module, "embed_dim", outShape[-1])
          else:
            batch = self.batchSize
            seq = entry.get("InputShapes", [[None, None]])[0][1] if (len(entry.get("InputShapes", [])) > 0) else 1
            embed = getattr(module, "embed_dim", 1)
          flops = 4 * batch * seq * seq * embed

        # Default: if none of the above matched, try to use activation element count as proxy.
        else:
          if (outShape is not None):
            numElems = 1
            for d in outShape:
              numElems *= d
            flops = numElems
      except Exception:
        flops = 0

      # Append per-layer flops entry to the list.
      perLayerFlops.append({
        "ModuleName"     : name,
        "LayerType"      : layerType,
        "EstimatedFLOPs" : flops,
        "OccurrenceIndex": occ
      })

      # Accumulate into the total flops counter.
      totalFlops += flops

    # Return flops summary including GFLOPs conversion.
    return {
      "TotalFLOPs"   : totalFlops,
      "PerLayerFLOPs": perLayerFlops,
      "TotalGFLOPs"  : round(totalFlops / 1e9, 4)
    }

  # Return the top-K layers by activation memory and parameter count.
  def _TopKMemoryLayers(self, k: int = 10) -> Dict[str, Any]:
    r'''
    Identify the top-k layers by activation memory and by local parameter count
    to help pinpoint memory hotspots in the model.

    Parameters:
      k (int):  Number of top entries to return for each category. Defaults to 10.

    Returns:
      Dict[str, Any]:  Dictionary containing "TopActivationLayers" and "TopParameterLayers".
    '''

    # Sort activation entries by ActivationMemoryBytes in descending order.
    sortedByActivation = sorted(self.activationMemoryList, key=lambda x: x["ActivationMemoryBytes"], reverse=True)

    # Select the top-k activation-heavy layers.
    topActivation = sortedByActivation[:k]

    # Build a mapping of module names to their local parameter counts.
    moduleParamCounts: Dict[str, int] = {}
    for name, module in self.model.named_modules():
      count = sum(p.numel() for p in module.parameters(recurse=False))
      if (count > 0):
        moduleParamCounts[name if (name != "") else module.__class__.__name__] = count

    # Sort modules by parameter count in descending order and pick top-k.
    sortedByParams = sorted(moduleParamCounts.items(), key=lambda x: x[1], reverse=True)[:k]

    # Return top-k lists as a dictionary.
    return {
      "TopActivationLayers": topActivation,
      "TopParameterLayers" : sortedByParams
    }

  # Estimate realistic sustained GFLOPS based on device availability and model type.
  def _EstimateRealisticGFLOPS(self, peakGFLOPS: float, isTransformer: bool = False) -> float:
    r'''
    Apply a heuristic scaling factor to a device's theoretical peak GFLOPS to
    estimate a more realistic sustained GFLOPS for the workload.

    Parameters:
      peakGFLOPS (float):  Theoretical peak GFLOPS of the device.
      isTransformer (bool):  Whether the model is transformer-like (affects factor).

    Returns:
      float:  Estimated sustained GFLOPS.
    '''

    # Decide on realism factor based on device.
    if ("cuda" in str(self.device).lower() or torch.cuda.is_available()):
      if (isTransformer):
        realism_factor = 0.20
      else:
        realism_factor = 0.40
    else:
      realism_factor = 0.15

    # Return scaled GFLOPS estimate.
    return peakGFLOPS * realism_factor

  # Profile memory for the model including parameters, activations, optimizer state, attention, and checkpointing.
  def ProfileModelMemory(
      self,
      optimizerType: str = "Adam",
      optimizerKwargs: Optional[Dict[str, Any]] = None,
      isTransformer: bool = False,
      sequenceLength: int = None,
      checkpointing: bool = False,
      checkpointSavingsFactor: float = 0.5,
      deviceFLOPSGFLOPS: Optional[float] = None,
      datasetSize: Optional[int] = None,
      trainingMultiplier: float = 3.0,
      runMicroBenchmark: bool = False
  ) -> Dict[str, Any]:
    r'''
    Perform a full memory and compute profile for the configured model. This
    runs a dummy forward pass (with hooks) to collect activation shapes,
    estimates parameter/buffer/optimizer/attention memory, and computes
    FLOPs-based performance estimates for training and inference.

    Parameters:
      optimizerType (str):  Optimizer used for estimating optimizer state memory.
      optimizerKwargs (Optional[Dict[str, Any]]):  Options passed to optimizer
        estimation (e.g., {"amsgrad": True}).
      isTransformer (bool):  If True, uses transformer-specific attention
        memory estimation; otherwise auto-detection may enable it.
      sequenceLength (int):  Required when isTransformer is True; sequence
        length used for attention memory estimation.
      checkpointing (bool):  Whether gradient checkpointing is enabled (reduces
        retained activation memory estimate).
      checkpointSavingsFactor (float):  Fraction of activation memory saved by
        checkpointing (0..1). Defaults to 0.5.
      deviceFLOPSGFLOPS (Optional[float]):  Optional device GFLOPS peak to use
        for performance estimates. If None a heuristic/default is chosen.
      datasetSize (Optional[int]):  Dataset size to estimate steps per epoch.
      trainingMultiplier (float):  Factor to scale forward GFLOPs to training
        GFLOPs (includes backward and optimizer work). Defaults to 3.0.
      runMicroBenchmark (bool):  If True attempts a small GEMM on the target
        device to empirically measure GFLOPS and refine timing estimates.

    Returns:
      Dict[str, Any]:  A comprehensive dictionary containing memory breakdowns (bytes and MB), layer-wise activations, top-K lists, FLOPs estimates, and performance estimates.
    '''

    # If a transformer is indicated but no sequence length is provided, raise an error.
    if (isTransformer and sequenceLength is None):
      raise ValueError("Sequence length must be provided for transformer models.")

    # Reset activation and layer info containers before profiling.
    self.activationMemoryList = []
    self.layerInfoList = []

    # Register forward hooks to capture activations during a dummy forward pass.
    self._RegisterForwardHooks()

    # Attempt to create dummy input on requested device and fall back to CPU if unavailable.
    # Determine dtype based on precision.
    dtype = torch.float16 if (self.precision == "FP16") else torch.float32
    try:
      dummyInput = torch.randn((self.batchSize,) + self.inputShape, device=self.device, dtype=dtype)
      actualDevice = self.device
    except Exception:
      dummyInput = torch.randn((self.batchSize,) + self.inputShape, device="cpu", dtype=dtype)
      actualDevice = "cpu"

    # Attempt to move the model to the requested device for the forward pass.
    try:
      self.model.to(self.device)
    except Exception:
      pass

    # Execute a forward pass under no-grad to collect activations.
    with torch.no_grad():
      _ = self.model(dummyInput)

    # Remove hooks after the forward pass to avoid side effects.
    for handle in getattr(self, "_hookHandles", []):
      try:
        handle.remove()
      except Exception:
        pass
    # Clear the hook handles list now that hooks are removed.
    self._hookHandles = []

    # Compute parameter counts using the internal helper.
    paramCounts = self._CountParameters()

    # Extract total and trainable parameter counts from the returned dictionary.
    totalParams = paramCounts["TotalParameters"]
    trainableParams = paramCounts["TrainableParameters"]

    # Compute buffer statistics for the model.
    bufferCounts = self._CountBuffers()

    # Compute parameter memory in bytes.
    parameterMemory = totalParams * self.bytesPerParam

    # Decide gradient and optimizer state bytes accounting for mixed precision common patterns.
    gradBytes = 4 if (self.precision == "FP16") else self.bytesPerParam
    optStateBytes = 4 if (self.precision == "FP16") else self.bytesPerParam

    # Compute gradient memory assuming one gradient tensor per trainable parameter using gradBytes.
    gradientMemory = trainableParams * gradBytes

    # Estimate optimizer state memory based on optimizer type and kwargs using optStateBytes.
    stateMemoryBase = self._EstimateOptimizerStateMemory(trainableParams, optimizerType, optimizerKwargs)
    # If the helper used self.bytesPerParam for calculation, adjust to optStateBytes.
    try:
      factor = optStateBytes / self.bytesPerParam
    except Exception:
      factor = 1.0
    optimizerMemory = int(stateMemoryBase * factor)

    # Sum activation memory recorded from hooks for all modules.
    totalActivationMemory = sum(item["ActivationMemoryBytes"] for item in self.activationMemoryList)

    # Estimate activation gradient memory conservatively as equal to activation memory before checkpointing.
    activationGradientMemory = totalActivationMemory

    # Estimate retained activations; adjust for checkpointing if enabled.
    if (checkpointing):
      retainedActivationMemory = int(totalActivationMemory * (1.0 - checkpointSavingsFactor))
    else:
      retainedActivationMemory = totalActivationMemory

    # Adjust activation gradient memory under checkpointing assumption conservatively.
    if (checkpointing):
      activationGradientMemory = retainedActivationMemory

    # Initialize attention memory to zero for non-transformer models.
    attentionMemory = 0

    # Auto-detect transformer presence if not explicitly requested.
    if (not isTransformer):
      for _, m in self.model.named_modules():
        if (isinstance(m, nn.MultiheadAttention)):
          isTransformer = True
          break

    # If transformer, attempt to infer sequence length and attention configuration.
    if (isTransformer):
      # Try to infer sequence length from recorded input shapes if not provided.
      seqLen = sequenceLength
      if (seqLen is None):
        for entry in self.activationMemoryList:
          for inpSh in entry.get("InputShapes", []):
            if (inpSh is None):
              continue
            if (len(inpSh) >= 2 and 1 < inpSh[1] < 10000):
              seqLen = inpSh[1]
              break
          if (seqLen is not None):
            break

      # Use a conservative default sequence length if inference failed.
      if (seqLen is None):
        seqLen = 128

      # Attempt to infer the number of heads from available attention modules.
      numHeads = 8
      for _, m in self.model.named_modules():
        if (isinstance(m, nn.MultiheadAttention)):
          numHeads = getattr(m, "num_heads", numHeads)

      # Count the MultiheadAttention occurrences as a proxy for number of layers.
      numLayers = sum(1 for _, m in self.model.named_modules() if (isinstance(m, nn.MultiheadAttention))) or 12

      # Estimate attention memory using the helper function.
      attentionMemory = self._EstimateAttentionMemory(seqLen, numHeads, numLayers)

    # Compute total memory required for training in bytes.
    totalTrainingMemory = (
        parameterMemory +
        gradientMemory +
        optimizerMemory +
        totalActivationMemory +
        activationGradientMemory +
        attentionMemory +
        bufferCounts["BufferMemoryBytes"]
    )

    # Compute approximate inference memory in bytes.
    totalInferenceMemory = parameterMemory + totalActivationMemory + attentionMemory + bufferCounts["BufferMemoryBytes"]

    # Compute FLOPs estimates using recorded activations.
    flops = self._EstimateFLOPs()

    # Compute top-K lists for memory and parameter heavy layers.
    topK = self._TopKMemoryLayers(k=10)

    # --- Performance estimates section.
    # Compute forward GFLOPs from flops estimate.
    forwardGFLOPs = flops.get("TotalGFLOPs", 0.0)

    # Inside ProfileModelMemory adjust training multiplier for transformers.
    if (trainingMultiplier == 3.0):
      if (isTransformer):
        trainingMultiplier = 3.5
      else:
        trainingMultiplier = 3.0

    # Compute training GFLOps per step using trainingMultiplier.
    trainingGFLOpsPerStep = forwardGFLOPs * trainingMultiplier

    # Get peak GFLOPS.
    peakGFLOPS = deviceFLOPSGFLOPS
    if (peakGFLOPS is None):
      if (torch.cuda.is_available() and ("cuda" in str(self.device).lower())):
        peakGFLOPS = 5000.0
      else:
        peakGFLOPS = 100.0

    # Convert to realistic sustained GFLOPS.
    deviceGFLOPS = self._EstimateRealisticGFLOPS(peakGFLOPS, isTransformer=isTransformer)

    # Avoid division by zero by clamping deviceGFLOPS.
    if (deviceGFLOPS <= 0):
      deviceGFLOPS = 1e-6

    # Compute time per training step in seconds with guards against zero FLOPs.
    if (trainingGFLOpsPerStep <= 0 or deviceGFLOPS <= 0):
      timePerTrainingStepSec = None
    else:
      timePerTrainingStepSec = trainingGFLOpsPerStep / deviceGFLOPS

    # Compute time per inference (forward) for the configured batch in seconds with guard.
    if (forwardGFLOPs <= 0 or deviceGFLOPS <= 0):
      timePerInferenceBatchSec = None
      timePerInferenceSampleSec = None
    else:
      timePerInferenceBatchSec = forwardGFLOPs / deviceGFLOPS
      timePerInferenceSampleSec = timePerInferenceBatchSec / max(1, self.batchSize)

    # Compute throughput numbers for training and inference with guards.
    if (timePerTrainingStepSec is None or timePerTrainingStepSec <= 0):
      trainingSamplesPerSecond = None
    else:
      trainingSamplesPerSecond = (1.0 / timePerTrainingStepSec) * max(1, self.batchSize)

    if (timePerInferenceSampleSec is None or timePerInferenceSampleSec <= 0):
      inferenceSamplesPerSecond = None
    else:
      inferenceSamplesPerSecond = 1.0 / timePerInferenceSampleSec

    # If datasetSize provided, compute steps per epoch and estimated epoch time.
    stepsPerEpoch = None
    timePerEpochSec = None
    if (datasetSize is not None):
      stepsPerEpoch = max(1, datasetSize // max(1, self.batchSize))
      if (timePerTrainingStepSec is None):
        timePerEpochSec = None
      else:
        timePerEpochSec = stepsPerEpoch * timePerTrainingStepSec

    # Optional micro-benchmark to estimate effective device GFLOPS (only if requested).
    microBenchmarkGFLOPS = None
    if (runMicroBenchmark):
      # Run a tiny GEMM to get an empirical timing and estimate GFLOPS.
      try:
        size = 1024
        a = torch.randn((size, size), device=self.device)
        b = torch.randn((size, size), device=self.device)
        torch.cuda.synchronize() if (torch.cuda.is_available() and "cuda" in str(self.device).lower()) else None
        t0 = time.time()
        c = a.matmul(b)
        torch.cuda.synchronize() if (torch.cuda.is_available() and "cuda" in str(self.device).lower()) else None
        t1 = time.time()
        elapsed = max(1e-9, t1 - t0)
        # FLOPs for matmul approx 2*N^3.
        flopEst = 2.0 * (size ** 3)
        microBenchmarkGFLOPS = (flopEst / 1e9) / elapsed
        # Replace deviceGFLOPS with empirical measure for more accurate timing.
        deviceGFLOPS = microBenchmarkGFLOPS
        # Recompute derived timings using empirical GFLOPS.
        timePerTrainingStepSec = trainingGFLOpsPerStep / deviceGFLOPS
        timePerInferenceBatchSec = forwardGFLOPs / deviceGFLOPS
        timePerInferenceSampleSec = timePerInferenceBatchSec / max(1, self.batchSize)
        trainingSamplesPerSecond = (1.0 / timePerTrainingStepSec) * max(1, self.batchSize)
        inferenceSamplesPerSecond = 1.0 / timePerInferenceSampleSec
        if (datasetSize is not None):
          timePerEpochSec = stepsPerEpoch * timePerTrainingStepSec
      except Exception:
        microBenchmarkGFLOPS = None

    # Assemble performance estimates dictionary.
    performanceEstimates = {
      "DeviceGFLOPSUsed"         : deviceGFLOPS,
      "MicroBenchmarkGFLOPS"     : microBenchmarkGFLOPS,
      "ForwardGFLOPsPerBatch"    : forwardGFLOPs,
      "TrainingGFLOPsPerStep"    : trainingGFLOpsPerStep,
      "TimePerTrainingStepSec"   : timePerTrainingStepSec,
      "TimePerInferenceBatchSec" : timePerInferenceBatchSec,
      "TimePerInferenceSampleSec": timePerInferenceSampleSec,
      "TrainingSamplesPerSecond" : trainingSamplesPerSecond,
      "InferenceSamplesPerSecond": inferenceSamplesPerSecond,
      "StepsPerEpoch"            : stepsPerEpoch,
      "TimePerEpochSec"          : timePerEpochSec
    }

    # Assemble the final memory profile dictionary using CamelCase keys.
    memoryProfile = {
      "ModelInfo"           : {
        "TotalParameters"       : totalParams,
        "TrainableParameters"   : trainableParams,
        "NonTrainableParameters": paramCounts["NonTrainableParameters"],
        "Precision"             : self.precision,
        "BatchSize"             : self.batchSize,
        "InputShape"            : self.inputShape,
        "Device"                : self.device
      },
      "MemoryBreakdownBytes": {
        "ParameterMemory"         : parameterMemory,
        "GradientMemory"          : gradientMemory,
        "OptimizerStateMemory"    : optimizerMemory,
        "ActivationMemory"        : totalActivationMemory,
        "ActivationGradientMemory": activationGradientMemory,
        "RetainedActivationMemory": retainedActivationMemory,
        "AttentionMemory"         : attentionMemory,
        "BufferMemory"            : bufferCounts["BufferMemoryBytes"],
        "TotalTrainingMemory"     : totalTrainingMemory,
        "TotalInferenceMemory"    : totalInferenceMemory
      },
      "MemoryBreakdownMB"   : {
        "ParameterMemory"         : round(parameterMemory / (1024 ** 2), 2),
        "GradientMemory"          : round(gradientMemory / (1024 ** 2), 2),
        "OptimizerStateMemory"    : round(optimizerMemory / (1024 ** 2), 2),
        "ActivationMemory"        : round(totalActivationMemory / (1024 ** 2), 2),
        "ActivationGradientMemory": round(activationGradientMemory / (1024 ** 2), 2),
        "RetainedActivationMemory": round(retainedActivationMemory / (1024 ** 2), 2),
        "AttentionMemory"         : round(attentionMemory / (1024 ** 2), 2),
        "BufferMemory"            : round(bufferCounts["BufferMemoryBytes"] / (1024 ** 2), 2),
        "TotalTrainingMemory"     : round(totalTrainingMemory / (1024 ** 2), 2),
        "TotalInferenceMemory"    : round(totalInferenceMemory / (1024 ** 2), 2)
      },
      "LayerWiseActivations": self.activationMemoryList,
      "TopKLayers"          : topK,
      "FLOPsEstimate"       : flops,
      "PerformanceEstimates": performanceEstimates,
      "TransformerSpecific" : {
        "IsTransformer"            : isTransformer,
        "SequenceLength"           : sequenceLength if (isTransformer) else None,
        "QuadraticComplexityFactor": sequenceLength ** 2 if (isTransformer and sequenceLength is not None) else None
      }
    }

    # Return the assembled memory profile dictionary.
    return memoryProfile

  # Helper to convert non-JSON-serializable objects into serializable forms.
  def _ToJsonSerializable(self, obj):
    r'''
    Convert common PyTorch / NumPy / Python objects to JSON-serializable
    representations. This is used as the `default` callable for json.dump.
    '''

    # Handle torch objects specially when torch is available.
    try:
      import numpy as _np
    except Exception:
      _np = None

    # Torch device (e.g., device(type="cuda", index=0)) -> string.
    # Torch dtype -> string.
    try:
      if (
          isinstance(obj, torch.device) or
          isinstance(obj, torch.dtype)
      ):
        return str(obj)
    except Exception:
      pass

    # Torch Size -> list.
    # Sets -> lists.
    try:
      if (
          isinstance(obj, torch.Size) or
          isinstance(obj, set)
      ):
        return list(obj)
    except Exception:
      pass

    # Torch Tensor -> small values as lists, otherwise summary dict.
    try:
      if (isinstance(obj, torch.Tensor)):
        # Move to CPU and detach to avoid GPU tensors in serialization.
        try:
          t = obj.detach().cpu()
        except Exception:
          t = obj
        numel = 0
        try:
          numel = int(t.numel())
        except Exception:
          numel = -1
        if (numel >= 0 and numel <= 16):
          try:
            return t.tolist()
          except Exception:
            return str(t)
        # For large tensors, return a compact summary to avoid huge JSON files.
        return {"__tensor__": True, "shape": list(t.shape), "dtype": str(t.dtype)}
    except Exception:
      pass

    # NumPy scalar types -> Python scalars.
    if (_np is not None):
      try:
        if (isinstance(obj, (_np.integer, _np.floating))):
          return obj.item()
        if (isinstance(obj, _np.ndarray)):
          if (obj.size <= 16):
            return obj.tolist()
          return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
      except Exception:
        pass

    # Bytes -> try decode else hex.
    try:
      if (isinstance(obj, (bytes, bytearray))):
        try:
          return obj.decode("utf-8")
        except Exception:
          return obj.hex()
    except Exception:
      pass

    # Fallback: try to convert __dict__ or use string repr.
    try:
      if (hasattr(obj, "__dict__")):
        d = {}
        for k, v in obj.__dict__.items():
          try:
            json.dumps(v)
            d[k] = v
          except TypeError:
            try:
              d[k] = str(v)
            except Exception:
              d[k] = None
        return d
    except Exception:
      pass

    # Last-resort: string representation.
    try:
      return str(obj)
    except Exception:
      return None

  # Save the produced memory profile into a JSON file.
  def SaveProfileToJSON(self, memoryProfile: Dict[str, Any], path: str) -> None:
    r'''
    Persist a memory profile dictionary to a JSON file path using utf-8
    encoding and pretty-print indentation for readability.

    Parameters:
      memoryProfile (Dict[str, Any]):  The profile produced by ProfileModelMemory.
      path (str):  Filesystem path where the JSON will be written.
    '''

    # Open the file and write the JSON dump with indentation using the
    # custom default serializer to handle torch/np types and other objects.
    with open(path, "w", encoding="utf-8") as f:
      json.dump(memoryProfile, f, indent=2, default=self._ToJsonSerializable)

  # Print a human readable memory report to stdout.
  def PrintMemoryReport(self, memoryProfile: Dict[str, Any]) -> None:
    r'''
    Nicely format and print a human-readable memory and performance report to
    standard output based on a profile produced by ProfileModelMemory.

    Parameters:
      memoryProfile (Dict[str, Any]):  The profile produced by ProfileModelMemory.
    '''

    # Extract the MB breakdown dictionary for easy access.
    mbBreakdown = memoryProfile["MemoryBreakdownMB"]

    # Print a header for the report.
    print("\n" + "=" * 70)
    print("MODEL MEMORY CONSUMPTION REPORT")
    print("=" * 70)

    # Print model configuration summary fields.
    modelInfo = memoryProfile["ModelInfo"]
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {modelInfo['TotalParameters']:,}")
    print(f"  Trainable Parameters: {modelInfo['TrainableParameters']:,}")
    print(f"  Precision: {modelInfo['Precision']}")
    print(f"  Batch Size: {modelInfo['BatchSize']}")
    print(f"  Input Shape: {modelInfo['InputShape']}")
    print(f"  Device: {modelInfo.get('Device', 'cpu')}")

    # Print training memory breakdown in MB.
    print(f"\nTraining Memory Consumption (MB):")
    print(f"  Parameters:       {mbBreakdown['ParameterMemory']:>8.2f} MB")
    print(f"  Gradients:        {mbBreakdown['GradientMemory']:>8.2f} MB")
    print(f"  Optimizer State:  {mbBreakdown['OptimizerStateMemory']:>8.2f} MB")
    print(f"  Activations:      {mbBreakdown['ActivationMemory']:>8.2f} MB")
    print(f"  Activation Grads: {mbBreakdown['ActivationGradientMemory']:>8.2f} MB")
    print(f"  Retained Activations (for backward): {mbBreakdown['RetainedActivationMemory']:>6.2f} MB")
    print(f"  Buffers (e.g., BN running stats):   {mbBreakdown['BufferMemory']:>6.2f} MB")

    # Print attention memory if the model is a transformer.
    if (memoryProfile["TransformerSpecific"]["IsTransformer"]):
      print(f"  Attention Matrices: {mbBreakdown['AttentionMemory']:>6.2f} MB")

    # Print the total training memory summary.
    print(f"  {'- ' * 12}")
    print(f"  TOTAL TRAINING:   {mbBreakdown['TotalTrainingMemory']:>8.2f} MB")

    # Print inference memory consumption summary.
    print(f"\nInference Memory Consumption (MB):")
    print(f"  Parameters + Activations + Buffers: {mbBreakdown['TotalInferenceMemory']:>8.2f} MB")

    # Print FLOPs estimates if available in the profile.
    flops = memoryProfile.get("FLOPsEstimate", {})
    if (flops):
      print(f"\nEstimated Compute (approx):")
      print(f"  Total FLOPs: {flops.get('TotalFLOPs', 0):,}")
      print(f"  Total GFLOPs: {flops.get('TotalGFLOPs', 0)} GFLOPs")

    # Print performance estimates if present.
    perf = memoryProfile.get("PerformanceEstimates", {})
    if (perf):
      print(f"\nPerformance Estimates:")
      print(f"  Device GFLOPS Used: {perf.get('DeviceGFLOPSUsed')}")
      if (perf.get("MicroBenchmarkGFLOPS") is not None):
        print(f"  Microbenchmark GFLOPS: {perf.get('MicroBenchmarkGFLOPS'):.2f}")
      print(f"  Forward GFLOPs per Batch: {perf.get('ForwardGFLOPsPerBatch')}")
      print(f"  Training GFLOPs per Step: {perf.get('TrainingGFLOPsPerStep')}")
      # Safely print time and throughput values, showing N/A when not available.
      tStep = perf.get("TimePerTrainingStepSec")
      if (tStep is None):
        print(f"  Time per Training Step: N/A")
      else:
        print(f"  Time per Training Step: {tStep:.6f} sec")

      if (perf.get("StepsPerEpoch") is not None):
        print(f"  Steps per Epoch: {perf.get('StepsPerEpoch')}")
        tEpoch = perf.get("TimePerEpochSec")
        if (tEpoch is None):
          print(f"  Time per Epoch: N/A")
        else:
          print(f"  Time per Epoch: {tEpoch:.2f} sec")

      tInfSample = perf.get("TimePerInferenceSampleSec")
      if (tInfSample is None):
        print(f"  Time per Inference Sample: N/A")
      else:
        print(f"  Time per Inference Sample: {tInfSample:.6f} sec")

      infThroughput = perf.get("InferenceSamplesPerSecond")
      if (infThroughput is None):
        print(f"  Inference Samples per Second: N/A")
      else:
        print(f"  Inference Samples per Second: {infThroughput:.2f}")

      trainThroughput = perf.get("TrainingSamplesPerSecond")
      if (trainThroughput is None):
        print(f"  Training Samples per Second: N/A")
      else:
        print(f"  Training Samples per Second: {trainThroughput:.2f}")

    # Print top-K lists for activation and parameter heavy layers.
    topK = memoryProfile.get("TopKLayers", {})
    if (topK):
      print(f"\nTop layers by Activation Memory:")
      for entry in topK.get("TopActivationLayers", [])[:10]:
        name = entry.get("ModuleName")
        bytesMB = round(entry.get("ActivationMemoryBytes", 0) / (1024 ** 2), 4)
        print(f"  {name:40s} {bytesMB:8.4f} MB  Shape: {entry.get('OutputShape')}")

      print(f"\nTop layers by Parameter Count:")
      for name, cnt in topK.get("TopParameterLayers", [])[:10]:
        print(f"  {name:40s} {cnt:,} params")

    # Print transformer complexity warning if applicable.
    if (memoryProfile["TransformerSpecific"]["IsTransformer"]):
      seqLen = memoryProfile["TransformerSpecific"]["SequenceLength"]
      quadFactor = memoryProfile["TransformerSpecific"]["QuadraticComplexityFactor"]
      print(f"\n\u26A0\uFE0F  Transformer Complexity Warning:")
      print(f"    Sequence Length: {seqLen}")
      print(f"    Quadratic Factor (N\u00B2): {quadFactor:,}")
      print(f"    Attention memory scales with O(N\u00B2) - doubling sequence length quadruples memory.")

    # In PrintMemoryReport, near the performance section print a caveat.
    print(f"\n\u2139\uFE0F  Performance estimates are approximations based on FLOPs and")
    print(f"    hardware utilization heuristics. Actual runtime may vary.")

    # Print a footer separator.
    print("=" * 70 + "\n")


# Example usage demonstration.
if (__name__ == "__main__"):
  # Import sample transformer model architecture.
  from torchvision.models import resnet18

  # Instantiate sample CNN model for profiling demonstration.
  sampleModel = resnet18()

  # Create profiler instance with standard ImageNet input dimensions.
  profiler = PyTorchModelMemoryProfiler(
    model=sampleModel,
    inputShape=(3, 224, 224),
    batchSize=1,
    precision="FP32",
    device="cuda",
  )

  # Execute comprehensive memory profiling.
  memoryReport = profiler.ProfileModelMemory(
    optimizerType="Adam",
    isTransformer=False,
    checkpointing=True,
    checkpointSavingsFactor=0.5,
    deviceFLOPSGFLOPS=5000.0,
    datasetSize=100000,
    trainingMultiplier=3.0,
    runMicroBenchmark=False
  )

  # Display human-readable memory consumption report.
  profiler.PrintMemoryReport(memoryReport)

  # Demonstrate transformer-specific profiling with hypothetical parameters.
  print("\n[Transformer Example]")
  transformerProfiler = PyTorchModelMemoryProfiler(
    model=sampleModel,
    inputShape=(3, 256, 256),
    batchSize=1,
    precision="FP32",
    device="cuda",
  )

  # Profile transformer with sequence length derived from patch embedding.
  transformerReport = transformerProfiler.ProfileModelMemory(
    optimizerType="Adam",
    optimizerKwargs={"amsgrad": True},
    isTransformer=True,
    sequenceLength=4096,
    checkpointing=True,
    checkpointSavingsFactor=0.5,
    deviceFLOPSGFLOPS=5000.0,
    datasetSize=100000,
    trainingMultiplier=3.0,
    runMicroBenchmark=False
  )

  # Display transformer memory report.
  transformerProfiler.PrintMemoryReport(transformerReport)
