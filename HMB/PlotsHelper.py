import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Optional


# Define a function to plot a heatmap from 2D data with flexible options.
def PlotHeatmap(
  data: np.ndarray,
  rowLabels: list[str],
  colLabels: list[str],
  title: str = "Heatmap",
  xlabel: str = "Column",
  ylabel: str = "Row",
  cmap: Any = "viridis",
  vmin: Optional[float] = None,
  vmax: Optional[float] = None,
  valueFormat: str = "{:.2f}",
  annotate: bool = True,
  fontSize: int = 7,
  figSize: Optional[tuple[float, float]] = None,
  colorbarLabel: Optional[str] = None,
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  dpi: int = 300,
  display: bool = True,
  colorbar: bool = True,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Plot a labeled heatmap from a 2D numeric array with flexible display and saving options.

  Parameters:
    data (numpy.ndarray): 2D numeric array to visualize (shape: [n_rows, n_cols]).
    rowLabels (list): Labels for the heatmap rows (y-axis). Length must match data.shape[0].
    colLabels (list): Labels for the heatmap columns (x-axis). Length must match data.shape[1].
    title (str): Plot title. Default is "Heatmap".
    xlabel (str): Label for x-axis. Default is "Column".
    ylabel (str): Label for y-axis. Default is "Row".
    cmap (matplotlib.colors.Colormap or str): Colormap to use for the heatmap. Default is "viridis".
    vmin (float or None): Minimum data value mapped to the colormap. Default is None.
    vmax (float or None): Maximum data value mapped to the colormap. Default is None.
    valueFormat (str): Format string used to annotate cell values (default: "{:.2f}").
    annotate (bool): Whether to annotate each cell with its numeric value. Default is True.
    fontSize (int): Font size used for annotations and axis labels. Default is 7.
    figSize (tuple): Figure size in inches. If None a size is inferred from label lengths.
    colorbarLabel (str or None): Label for the colorbar if shown.
    save (bool): Whether to save the plot. Default is False.
    savePath (str or pathlib.Path or None): Path to save the figure. If provided and `save` is False
      the function will still save to this path (legacy behavior).
    fileName (str or None): Alternative filename to save the figure when `save` is True.
    dpi (int): DPI for saving the figure. Default is 300.
    display (bool): Whether to display the plot using plt.show(). Default is True.
    colorbar (bool): Whether to show a colorbar. Default is True.
    returnFig (bool): Whether to return the matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - When saving, a PNG sibling file is written alongside the requested format where possible.
    - The function tolerates NaN/Inf in the data: those cells are not annotated.
    - Backwards compatibility: if `savePath` is provided without `save`, the function
      will still save to `savePath` (behavior maintained from prior versions).

  Examples
  --------
  .. code-block:: python

    import numpy as np
    from HMB.PlotsHelper import PlotHeatmap

    data = np.random.rand(4, 6)
    PlotHeatmap(
      data,
      rowLabels=[f"R{i}" for i in range(4)],
      colLabels=[f"C{j}" for j in range(6)],
      title="Example Heatmap",
      annotate=True,
      display=True,
    )
  '''

  # Compute default figure size when not provided.
  if (figSize is None):
    # Set figure width and height based on label counts.
    figSize = (
      max(6.0, float(len(colLabels)) * 0.6),
      max(4.0, float(len(rowLabels)) * 0.4)
    )

  # Create a new matplotlib Figure and Axes for the heatmap.
  fig, ax = plt.subplots(figsize=figSize)

  # Display the data array as an image on the Axes with the provided colormap.
  im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

  # Set y-axis tick positions based on number of rows.
  ax.set_yticks(np.arange(len(rowLabels)))

  # Set y-axis tick labels from the provided rowLabels.
  ax.set_yticklabels(rowLabels)

  # Set x-axis tick positions based on number of columns.
  ax.set_xticks(np.arange(len(colLabels)))

  # Set x-axis tick labels from the provided colLabels and rotate for readability.
  ax.set_xticklabels(colLabels, rotation=45, ha="right")

  # Set the x-axis label text.
  ax.set_xlabel(xlabel)

  # Set the y-axis label text.
  ax.set_ylabel(ylabel)

  # Set the title if provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title)

  # Optionally add a colorbar to the Figure.
  cbar = None
  if (colorbar):
    # Create a colorbar for the image using the Figure and Axes.
    cbar = fig.colorbar(im, ax=ax)
    # Set the colorbar label if provided.
    if (colorbarLabel is not None):
      # Assign the label text to the colorbar.
      cbar.set_label(colorbarLabel)

  # Optionally annotate each cell with its numeric value.
  if (annotate):
    # Compute the maximum of the data for thresholding text color.
    try:
      dataMax = np.nanmax(data)
    except Exception:
      # If computing the maximum fails, set dataMax to None.
      dataMax = None
    # Compute a threshold to decide text color for contrast.
    thresh = (dataMax / 2.0) if (dataMax is not None and dataMax != 0) else None

    # Iterate over rows to annotate cell values.
    for i in range(data.shape[0]):
      # Iterate over columns for each row.
      for j in range(data.shape[1]):
        # Read the cell value from the data array.
        val = data[i, j]
        # Skip annotation for NaN or infinite values.
        if (not (np.isnan(val) or np.isinf(val))):
          # Default text color for annotations.
          color = "white"
          # If threshold is available, choose text color for contrast.
          if (thresh is not None):
            try:
              # Use white text when value is above threshold, otherwise black.
              color = "white" if val > thresh else "black"
            except Exception:
              # Fallback to white in case of any error.
              color = "white"

          # Draw the annotation text at the center of the cell.
          ax.text(
            j, i,
            valueFormat.format(val),
            ha="center", va="center",
            color=color,
            fontsize=fontSize
          )

  # Tighten the layout of the Figure to avoid clipping labels.
  fig.tight_layout()

  # Optionally save the Figure to disk when requested.
  if (save):
    # Initialize the target path variable.
    targetPath: Optional[Path] = None
    # Prefer explicit savePath if provided.
    if (savePath is not None):
      # Convert savePath to a Path object.
      targetPath = Path(savePath)
    elif (fileName is not None):
      # Convert fileName to a Path object.
      targetPath = Path(fileName)

    # If a target path was determined, create directories and save.
    if (targetPath is not None):
      # Ensure parent directories exist.
      targetPath.parent.mkdir(parents=True, exist_ok=True)
      # Determine file extension for additional PNG save.
      ext = targetPath.suffix.lower().lstrip('.')
      try:
        # Attempt to save the Figure to the requested path.
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print an error message if saving fails.
        print(f"Error saving heatmap: {e}")
      # Also write a PNG copy when the primary extension is not PNG.
      if (ext != "png"):
        try:
          # Save a PNG sibling file.
          fig.savefig(str(targetPath.with_suffix('.png')), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore PNG save errors silently.
          pass

  # Backwards compatibility: save when savePath provided even if save is False.
  if (savePath is not None) and (not save) and (fileName is None):
    try:
      # Convert the legacy savePath to a Path object.
      pathObj = Path(savePath)
      # Ensure parent directories exist.
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      # Save the Figure to the legacy path.
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      # Inform the user where the heatmap was saved.
      print(f"Heatmap saved to: {pathObj}")
    except Exception:
      # Ignore save errors in legacy path behavior.
      pass

  # Optionally display the Figure using matplotlib's show.
  if (display):
    # Show the Figure on screen.
    plt.show()

  # Close the Figure to release resources.
  plt.close(fig)

  # Optionally return the created Figure object.
  if (returnFig):
    # Return the Figure instance to the caller.
    return fig

  # Return None when not returning the Figure.
  return None


# Define a function to plot a bar chart from values and labels.
def PlotBarChart(
  values: list[float],
  labels: list[str],
  title: str = "",
  ylabel: str = "",
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  color: Any = "tab:blue",
  colors: Optional[list[str]] = None,
  alpha: float = 0.85,
  dpi: int = 300,
  display: bool = True,
  annotate: bool = False,
  annotateFormat: str = "{:.2f}",
  annotateFontSize: Optional[int] = None,
  fontSize: int = 10,
  figSize: Optional[tuple[float, float]] = None,
  rotation: float = 45,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Create a bar chart from values and labels with options to annotate, save, display and return the Figure.

  Parameters:
    values (list): Numeric heights for each bar.
    labels (list): Labels for each bar (x-axis). Must have the same length as `values`.
    title (str): Plot title. Default is empty.
    ylabel (str): Label for the y-axis.
    save (bool): Whether to save the plot. Default is False.
    savePath (str or pathlib.Path or None): Path to save the figure. Kept for backward compatibility;
      when provided and `save` is False the function will still save to this path.
    fileName (str or None): Alternative filename to save the figure when `save` is True.
    color (str or color spec): Single color for all bars (default: "tab:blue").
    colors (list or None): Per-bar color list. When provided it overrides `color`.
    alpha (float): Alpha (opacity) applied to bars. Default is 0.85.
    dpi (int): DPI for saving the figure. Default is 300.
    display (bool): Whether to display the plot using plt.show(). Default is True.
    annotate (bool): Whether to annotate each bar with its numeric value. Default is False.
    annotateFormat (str): Format string used for bar annotations (default: "{:.2f}").
    annotateFontSize (int or None): Font size for annotations. If None a size is derived from `fontSize`.
    fontSize (int): Base font size for axis labels and ticks. Default is 10.
    figSize (tuple or None): Figure size in inches. When None a width is inferred from number of bars.
    rotation (float): Rotation applied to x tick labels (degrees). Default is 45.
    returnFig (bool): Whether to return the matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The matplotlib figure object if returnFig is True, otherwise None.

  Notes:
    - When saving, a PNG sibling file is written alongside other formats where possible.
    - Backwards compatibility: if `savePath` is provided without `save`, the function
      will still save to `savePath` (preserving legacy behavior).

  Examples
  --------
  .. code-block:: python

    from HMB.PlotsHelper import PlotBarChart
    PlotBarChart([1.2, 3.4, 2.1], ["A", "B", "C"], title="Counts", annotate=True)
  '''

  # Validate that values and labels lengths match.
  if (len(values) != len(labels)):
    # Raise when lengths mismatch.
    raise ValueError("Length of 'values' must match 'labels'.")

  # Return early when there are no labels to plot.
  if (len(labels) == 0):
    return None

  # Infer a default figure size when not provided.
  if (figSize is None):
    # Compute width based on number of labels.
    figWidth = max(6.0, float(len(labels)) * 0.5)
    # Set figSize tuple.
    figSize = (figWidth, 4.0)

  # Create a new Figure and Axes for the bar chart.
  fig, ax = plt.subplots(figsize=figSize)

  # Decide bar colors: prefer per-bar list when provided.
  if (colors is not None):
    # Use the provided per-bar colors.
    barColors = colors
  else:
    # Use a single color specification for all bars.
    barColors = color

  # Draw the bars on the Axes.
  ax.bar(range(len(labels)), values, color=barColors, alpha=alpha)

  # Set x tick positions for each label.
  ax.set_xticks(range(len(labels)))

  # Set x tick labels with rotation and font size.
  ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontSize)

  # Set the y-axis label text.
  ax.set_ylabel(ylabel, fontsize=fontSize)

  # Set the title when provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title, fontsize=fontSize)

  # Tighten layout to avoid clipping.
  fig.tight_layout()

  # Optionally annotate each bar with its value.
  if (annotate):
    # Derive a sensible annotation font size when not explicitly provided.
    if (annotateFontSize is None):
      # Compute annotation font size relative to base font size.
      annotateFontSize = max(8, int(fontSize * 0.9))
    # Iterate over bars to place annotation texts.
    for i, v in enumerate(values):
      try:
        # Format the annotation text using the provided format.
        label = annotateFormat.format(v)
      except Exception:
        # Fallback to plain string conversion on failure.
        label = str(v)
      # Place the annotation text above the bar.
      ax.text(i, v, label, ha="center", va="bottom", fontsize=annotateFontSize)

  # Optionally save the bar chart to disk when requested.
  if (save):
    # Initialize target path variable.
    targetPath: Optional[Path] = None
    # Prefer explicit savePath when provided.
    if (savePath is not None):
      # Convert savePath to a Path object.
      targetPath = Path(savePath)
    elif (fileName is not None):
      # Convert fileName to a Path object.
      targetPath = Path(fileName)

    # If a target path was determined, persist the figure.
    if (targetPath is not None):
      # Create parent directories if required.
      targetPath.parent.mkdir(parents=True, exist_ok=True)
      try:
        # Save the Figure to the requested path.
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print an error message if saving fails.
        print(f"Error saving bar chart: {e}")
      # Also save a PNG copy when the primary extension is not PNG.
      if (targetPath.suffix.lower().lstrip('.') != "png"):
        try:
          # Save a PNG sibling file.
          fig.savefig(str(targetPath.with_suffix('.png')), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore errors when saving the PNG copy.
          pass

  # Legacy behavior: if savePath provided but save is False, still save for compatibility.
  if ((savePath is not None) and (not save) and (fileName is None)):
    try:
      # Convert the legacy savePath to a Path object.
      pathObj = Path(savePath)
      # Ensure parent directories exist.
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      # Save the Figure to the legacy path.
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      # Inform the user where the bar chart was saved.
      print(f"Bar chart saved to: {pathObj}")
    except Exception:
      # Ignore legacy save errors.
      pass

  # Optionally display the Figure on screen.
  if (display):
    # Show the Figure using matplotlib.
    plt.show()

  # Close the Figure to release memory.
  plt.close(fig)

  # Optionally return the Figure instance to the caller.
  if (returnFig):
    # Return the Figure object.
    return fig

  # Return None when not returning the Figure.
  return None
