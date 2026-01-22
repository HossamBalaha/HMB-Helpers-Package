import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Optional
from matplotlib import colors as mcolors


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
      ext = targetPath.suffix.lower().lstrip(".")
      try:
        # Attempt to save the Figure to the requested path.
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print an error message if saving fails.
        print(f"Error saving heatmap: {e}")
      # Also write a PNG sibling file when the primary extension is not PNG.
      if (ext != "png"):
        try:
          # Save a PNG sibling file.
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
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
  annotateFormat: str = "{:.4f}",
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
    # Nothing to plot; return None to indicate no figure created.
    return None

  # Infer a default figure size when not provided.
  if (figSize is None):
    # Compute width based on number of labels.
    figWidth = max(6.0, float(len(labels)) * 0.5)
    # Set figSize tuple.
    figSize = (figWidth, 4.0)

  # Create a new Figure and Axes for the bar chart.
  fig, ax = plt.subplots(figsize=figSize)

  # Add dashed grid lines along the y-axis for readability.
  ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.6)
  # Keep the dashed grid lines behind the bars.
  ax.set_axisbelow(True)

  # Decide bar colors: prefer per-bar list when provided.
  if (colors is not None):
    # Use the provided per-bar colors.
    barColors = colors
  else:
    # Use a single color specification for all bars.
    barColors = color

  # Draw the bars on the Axes.
  bars = ax.bar(range(len(labels)), values, color=barColors, alpha=alpha)

  # Keep the bar edges sharp.
  ax.bar(range(len(labels)), values, color=barColors, alpha=alpha, edgecolor="black", linewidth=0.7)

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

  # Optionally annotate each bar with its value placed in the middle and rotated 45 degrees.
  if (annotate):
    # Derive a sensible annotation font size when not explicitly provided.
    if (annotateFontSize is None):
      # Compute annotation font size relative to base font size.
      annotateFontSize = max(8, int(fontSize * 0.9))

    # Iterate over bars to place annotation texts at the bar center with rotation.
    for idx, bar in enumerate(bars):
      # Safely format the annotation label.
      try:
        label = annotateFormat.format(values[idx])
      except Exception:
        label = str(values[idx])

      # Get bar height for vertical placement (works for positive and negative bars).
      barHeight = float(bar.get_height())

      # Compute bar bottom/top and center (handles both positive and negative values).
      barBottom = float(bar.get_y())
      barTop = barBottom + float(bar.get_height())
      barCenter = barBottom + (barHeight / 2.0)

      # Determine axis span to decide whether there's room inside the bar for the label.
      yMin, yMax = ax.get_ylim()
      ySpan = max(1e-6, (yMax - yMin))

      # If the bar height is large enough relative to the axis span, put the label inside (vertically centered).
      # Otherwise place it slightly above the bar top and expand y-limits if needed.
      minRatioForInside = 0.08
      if (abs(barHeight) >= (ySpan * minRatioForInside)):
        yPos = barCenter
        va = "center"
        clipOn = True
      else:
        offset = ySpan * 0.02
        # For positive bars place above top, for negative bars place below bottom.
        if (barHeight >= 0):
          yPos = barTop + offset
          va = "bottom"
          # Expand y-axis top if label would be out of bounds.
          if (yPos > yMax):
            ax.set_ylim(yMin, yPos + ySpan * 0.05)
        else:
          # Negative bar: put label below the bar.
          yPos = barTop - offset
          va = "top"
          if yPos < yMin:
            ax.set_ylim(yPos - ySpan * 0.05, yMax)
        clipOn = False

      # Determine x position centered on the bar.
      xPos = bar.get_x() + bar.get_width() / 2.0

      # Place the annotation. Use different vertical alignment and clipping depending on placement.
      ax.text(
        xPos, yPos, label,
        ha="center",
        va=va,
        fontsize=annotateFontSize,
        rotation=45,
        color="black",
        clip_on=clipOn,
        zorder=10,
      )

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
      if (targetPath.suffix.lower().lstrip(".") != "png"):
        try:
          # Save a PNG sibling file.
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
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


# Additional generic plot helpers.
def PlotHorizontalBarChart(
  values: list[float],
  labels: list[str],
  title: str = "",
  xlabel: str = "",
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  color: Any = "tab:blue",
  colors: Optional[list[str]] = None,
  alpha: float = 0.85,
  dpi: int = 300,
  display: bool = True,
  annotate: bool = False,
  annotateFormat: str = "{:.4f}",
  annotateFontSize: Optional[int] = None,
  fontSize: int = 10,
  figSize: Optional[tuple[float, float]] = None,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Create a horizontal bar chart (bars extend along the x-axis) with labeled y-axis.

  This helper creates a horizontal bar chart suitable for categorical counts or scores.
  It mirrors the options and backwards-compatible saving behavior of `PlotBarChart`.

  Parameters:
    values (list[float]): Numeric lengths for each horizontal bar.
    labels (list[str]): Labels for each bar shown on the y-axis. Length must match `values`.
    title (str): Plot title. Default is empty.
    xlabel (str): Label for the x-axis. Default is empty.
    save (bool): Whether to save the plot to disk. Default is False.
    savePath (str|Path|None): Backwards-compatible path to save the figure. When provided and
      `save` is False the function still saves to this location (legacy behavior).
    fileName (str|None): Alternative file path used when `save` is True.
    color (Any): Single color specification for all bars (default "tab:blue").
    colors (list[str]|None): Per-bar color list that overrides `color` when provided.
    alpha (float): Opacity applied to bars. Default is 0.85.
    dpi (int): DPI for saved figures. Default is 300.
    display (bool): Whether to call `plt.show()` after drawing. Default is True.
    annotate (bool): Whether to annotate each bar with its numeric value. Default is False.
    annotateFormat (str): Format string used for annotations. Default is "{:.2f}".
    annotateFontSize (int|None): Font size for bar annotations. If None a sensible size is derived.
    fontSize (int): Base font size used for axis labels and ticks. Default is 10.
    figSize (tuple|None): Figure size in inches. If None a default is inferred from number of labels.
    returnFig (bool): Whether to return the created matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The created Figure when `returnFig` is True, otherwise None.

  Notes:
    - When saving, a PNG sibling file is written alongside other formats when possible.
    - Backwards compatibility: if `savePath` is provided without `save`, the function will still save
      to `savePath` (preserving legacy behavior used elsewhere in this module).

  Examples
  --------
  .. code-block:: python

    from HMB.PlotsHelper import PlotHorizontalBarChart
    PlotHorizontalBarChart([10, 5, 7], ["A","B","C"], title="Counts", save=True, fileName="counts.pdf")
  '''

  # Validate that the provided lists have matching length.
  if (len(values) != len(labels)):
    # Raise an informative exception when lengths mismatch.
    raise ValueError("Length of 'values' must match 'labels'.")

  # Return early when there are no labels to plot.
  if (len(labels) == 0):
    # Nothing to plot; return None to indicate no figure created.
    return None

  # Compute a default figure size when not provided.
  if (figSize is None):
    # Compute height based on number of labels for readability.
    figHeight = max(4.0, float(len(labels)) * 0.35)
    # Set the computed figure size tuple.
    figSize = (8.0, figHeight)

  # Create a matplotlib Figure and Axes for drawing.
  fig, ax = plt.subplots(figsize=figSize)

  # Decide bar colors preferring per-bar list when provided.
  barColors = (colors if (colors is not None) else color)

  # Build y positions for the horizontal bars.
  yPositions = range(len(labels))

  # Draw the horizontal bars on the Axes.
  ax.barh(yPositions, values, color=barColors, alpha=alpha)

  # Set y tick positions to align with bars.
  ax.set_yticks(yPositions)

  # Set y tick labels using provided labels.
  ax.set_yticklabels(labels, fontsize=fontSize)

  # Set the x-axis label text.
  ax.set_xlabel(xlabel, fontsize=fontSize)

  # Optionally set the plot title when provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title, fontsize=fontSize)

  # Tighten layout to avoid clipping labels.
  fig.tight_layout()

  # Optionally annotate each bar with its numeric value.
  if (annotate):
    # Compute a sensible annotation font size when not provided.
    if (annotateFontSize is None):
      # Derive annotation font size from base font size.
      annotateFontSize = max(8, int(fontSize * 0.9))
    # Iterate over bars to place annotation texts.
    for i, v in enumerate(values):
      # Format the annotation text safely.
      try:
        label = annotateFormat.format(v)
      except Exception:
        label = str(v)
      # Place the annotation text adjacent to the bar value.
      ax.text(
        v, i, f" {label}", va="center", fontsize=annotateFontSize
      )

  # Optionally save the figure to disk when requested.
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
        print(f"Error saving horizontal bar chart: {e}")
      # Also save a PNG copy when the primary extension is not PNG.
      if (targetPath.suffix.lower().lstrip(".") != "png"):
        try:
          # Save a PNG sibling file.
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore errors when saving the PNG copy.
          pass

  # Legacy behavior: if savePath provided but save flag is False still persist for compatibility.
  if ((savePath is not None) and (not save) and (fileName is None)):
    try:
      # Convert the legacy savePath to a Path object.
      pathObj = Path(savePath)
      # Ensure parent directories exist.
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      # Save the Figure to the legacy path.
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      # Inform the user where the horizontal bar chart was saved.
      print(f"Horizontal bar chart saved to: {pathObj}")
    except Exception:
      # Ignore legacy save errors silently.
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

  # Default return when nothing to return.
  return None


def PlotPieChart(
  values: list[float],
  labels: list[str],
  title: str = "",
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  autopct: Optional[str] = "%1.1f%%",
  explode: Optional[list[float]] = None,
  colors: Optional[list[str]] = None,
  dpi: int = 300,
  display: bool = True,
  fontSize: int = 10,
  figSize: Optional[tuple[float, float]] = None,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Create a pie chart from categorical values and labels.

  This helper draws a pie chart and a right-side legend for readability. The chart
  accepts options for percentage annotations, slice explosion and custom colors.

  Parameters:
    values (list[float]): Numeric values for pie slices. Length must match `labels`.
    labels (list[str]): Labels for each slice.
    title (str): Plot title. Default is empty.
    save (bool): Whether to save the plot to disk. Default is False.
    savePath (str|Path|None): Backwards-compatible save path used when provided.
    fileName (str|None): Alternative file path used when `save` is True.
    autopct (str|None): Format string for slice percentage annotation (e.g. "%1.1f%%").
    explode (list[float]|None): Per-slice offset fractions for emphasizing slices.
    colors (list[str]|None): Per-slice colors.
    dpi (int): DPI for saved figures. Default is 300.
    display (bool): Whether to display the figure with `plt.show()`. Default is True.
    fontSize (int): Font size used in the legend and title. Default is 10.
    figSize (tuple|None): Figure size in inches. Default is a square figure.
    returnFig (bool): Whether to return the created matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The created Figure when `returnFig` is True, otherwise None.

  Notes:
    - The function places labels in a legend to the right to keep the pie area uncluttered.
    - Backwards-compatible saving behavior is preserved (see `savePath` semantics).

  Examples
  --------
  .. code-block:: python

    from HMB.PlotsHelper import PlotPieChart
    PlotPieChart([30, 70], ["CatA", "CatB"], title="Distribution", save=True, fileName="dist.png")
  '''

  # Validate that the lengths of values and labels match.
  if (len(values) != len(labels)):
    # Raise an informative exception for mismatched lengths.
    raise ValueError("Length of 'values' must match 'labels'.")

  # Return early when there are no labels to plot.
  if (len(labels) == 0):
    # Nothing to plot; return None.
    return None

  # Determine a default figure size when not provided.
  if (figSize is None):
    # Use a square figure for pie charts.
    figSize = (6.0, 6.0)

  # Create a new Figure and Axes for the pie chart.
  fig, ax = plt.subplots(figsize=figSize)

  # Draw the pie chart wedges without labels (legend will show names).
  wedges_texts = ax.pie(
    values,
    labels=None,
    autopct=autopct,
    explode=explode,
    colors=colors,
    startangle=90,
  )

  # Ensure the pie is rendered as a circle.
  ax.axis("equal")

  # Add a legend with labels placed to the right for readability.
  ax.legend(wedges_texts[0], labels, title=None, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontSize)

  # Optionally set the title when provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title, fontsize=fontSize)

  # Tighten layout to avoid clipping the legend.
  fig.tight_layout()

  # Persist the figure to disk when requested via save flag.
  if (save):
    # Initialize the target path variable.
    targetPath: Optional[Path] = None
    # Prefer explicit savePath when provided.
    if (savePath is not None):
      # Convert savePath to a Path object.
      targetPath = Path(savePath)
    elif (fileName is not None):
      # Convert fileName to a Path object.
      targetPath = Path(fileName)
    # Save the figure when a target path was determined.
    if (targetPath is not None):
      # Ensure parent directories exist.
      targetPath.parent.mkdir(parents=True, exist_ok=True)
      # Attempt to save the Figure to the requested path.
      try:
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print an error message when saving fails.
        print(f"Error saving pie chart: {e}")
      # Also write a PNG sibling file when the primary extension is not PNG.
      if (targetPath.suffix.lower().lstrip(".") != "png"):
        try:
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore PNG save errors silently.
          pass

  # Legacy behavior: save when savePath provided even if save flag is False.
  if ((savePath is not None) and (not save) and (fileName is None)):
    # Attempt to save to the legacy path for compatibility.
    try:
      pathObj = Path(savePath)
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      print(f"Pie chart saved to: {pathObj}")
    except Exception:
      # Ignore legacy save errors silently.
      pass

  # Display the figure interactively when requested.
  if (display):
    # Show the matplotlib figure.
    plt.show()

  # Close the figure to release memory.
  plt.close(fig)

  # Optionally return the Figure when requested.
  if (returnFig):
    # Return the Figure instance to the caller.
    return fig

  # Default return when no figure is returned.
  return None


def PlotLineChart(
  x: Optional[list[float]],
  y: list[float],
  title: str = "",
  xlabel: str = "",
  ylabel: str = "",
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  dpi: int = 300,
  display: bool = True,
  annotate: bool = False,
  annotateFormat: str = "{:.2f}",
  fontSize: int = 10,
  figSize: Optional[tuple[float, float]] = None,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Plot a simple line chart connecting (x, y) points.

  This helper supports optional annotation of points and standard saving/display
  options consistent with other plot helpers in this module.

  Parameters:
    x (list[float]|None): X coordinates for points. When None the indices of `y` are used.
    y (list[float]): Y coordinates for points.
    title (str): Plot title. Default is empty.
    xlabel (str): X-axis label. Default is empty.
    ylabel (str): Y-axis label. Default is empty.
    save (bool): Whether to save the plot to disk. Default is False.
    savePath (str|Path|None): Backwards-compatible save path used when provided.
    fileName (str|None): Alternative file path used when `save` is True.
    dpi (int): DPI for saved figures. Default is 300.
    display (bool): Whether to display the figure with `plt.show()`. Default is True.
    annotate (bool): Whether to annotate each point with its value. Default is False.
    annotateFormat (str): Format string used for annotations. Default is "{:.2f}".
    fontSize (int): Font size used for labels and annotations. Default is 10.
    figSize (tuple|None): Figure size in inches.
    returnFig (bool): Whether to return the created matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The created Figure when `returnFig` is True, otherwise None.

  Notes:
    - When `x` is None the function uses sequential integer indices for the x-axis.
    - Backwards-compatible saving behavior is preserved.

  Examples
  --------
  .. code-block:: python

    from HMB.PlotsHelper import PlotLineChart
    PlotLineChart(None, [1, 3, 2, 5], title="Signal", annotate=True)
  '''

  # Use index sequence for x when None is provided.
  if (x is None):
    # Create a range index matching the length of y.
    x = list(range(len(y)))

  # Validate that x and y lengths match.
  if (len(x) != len(y)):
    # Raise when the provided x and y lengths differ.
    raise ValueError("Length of \"x\" must match \"y\".")

  # Return early when there is no data to plot.
  if (len(y) == 0):
    # Nothing to plot; return None.
    return None

  # Determine default figure size when not given.
  if (figSize is None):
    # Use a standard wide figure for line plots.
    figSize = (8.0, 4.0)

  # Create the Figure and Axis for plotting.
  fig, ax = plt.subplots(figsize=figSize)

  # Plot the line with markers.
  ax.plot(x, y, marker="o")

  # Set x-axis label text.
  ax.set_xlabel(xlabel, fontsize=fontSize)

  # Set y-axis label text.
  ax.set_ylabel(ylabel, fontsize=fontSize)

  # Optionally set the title when provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title, fontsize=fontSize)

  # Tighten layout to avoid clipping.
  fig.tight_layout()

  # Optionally annotate individual points with their values.
  if (annotate):
    # Iterate through points and add annotations.
    for xi, yi in zip(x, y):
      # Format the annotation text safely.
      try:
        txt = annotateFormat.format(yi)
      except Exception:
        txt = str(yi)
      # Place the annotation slightly above the point.
      ax.annotate(
        txt, (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center",
        fontsize=max(8, int(fontSize * 0.9))
      )

  # Persist the figure to disk when requested.
  if (save):
    # Determine the target path object for saving.
    targetPath: Optional[Path] = None
    # Prefer explicit savePath when provided.
    if (savePath is not None):
      # Convert savePath to a Path object.
      targetPath = Path(savePath)
    elif (fileName is not None):
      # Convert fileName to a Path object.
      targetPath = Path(fileName)
    # Save the figure when a target path exists.
    if (targetPath is not None):
      # Ensure parent directories exist for the target path.
      targetPath.parent.mkdir(parents=True, exist_ok=True)
      # Attempt to write the figure to the requested path.
      try:
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print a helpful message on save failure.
        print(f"Error saving line chart: {e}")
      # Also attempt a PNG sibling when primary extension is not PNG.
      if (targetPath.suffix.lower().lstrip(".") != "png"):
        try:
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore PNG sibling save errors silently.
          pass

  # Legacy behavior: save when savePath provided even when save flag is False.
  if ((savePath is not None) and (not save) and (fileName is None)):
    # Attempt to persist to legacy savePath for compatibility.
    try:
      pathObj = Path(savePath)
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      print(f"Line chart saved to: {pathObj}")
    except Exception:
      # Ignore legacy save errors silently.
      pass

  # Display the figure when requested.
  if (display):
    # Show the matplotlib figure.
    plt.show()

  # Close the figure to release memory.
  plt.close(fig)

  # Optionally return the created Figure object.
  if (returnFig):
    # Return the Figure instance.
    return fig

  # Default return when nothing to return.
  return None


def PlotHistogram(
  data: list[float],
  bins: int = 20,
  title: str = "",
  xlabel: str = "",
  ylabel: str = "Frequency",
  save: bool = False,
  savePath: Optional[Path | str] = None,
  fileName: Optional[str] = None,
  dpi: int = 300,
  display: bool = True,
  fontSize: int = 10,
  figSize: Optional[tuple[float, float]] = None,
  returnFig: bool = False,
) -> Optional[plt.Figure]:
  r'''
  Plot a histogram for 1D numeric data.

  This helper draws a histogram with configurable bin count and axis labels. It
  follows the same saving and display semantics used elsewhere in this module.

  Parameters:
    data (list[float]): 1D numeric sequence to bin and plot.
    bins (int): Number of histogram bins. Default is 20.
    title (str): Plot title. Default is empty.
    xlabel (str): X-axis label. Default is empty.
    ylabel (str): Y-axis label. Default is "Frequency".
    save (bool): Whether to save the plot to disk. Default is False.
    savePath (str|Path|None): Backwards-compatible save path used when provided.
    fileName (str|None): Alternative file path used when `save` is True.
    dpi (int): DPI for saved figures. Default is 300.
    display (bool): Whether to display the figure with `plt.show()`. Default is True.
    fontSize (int): Font size used for axis labels. Default is 10.
    figSize (tuple|None): Figure size in inches. Default is a standard wide figure.
    returnFig (bool): Whether to return the created matplotlib Figure object. Default is False.

  Returns:
    matplotlib.figure.Figure or None: The created Figure when `returnFig` is True, otherwise None.

  Notes:
    - Backwards-compatible saving behavior is preserved.

  Examples
  --------
  .. code-block:: python

    from HMB.PlotsHelper import PlotHistogram
    PlotHistogram([0.1,0.2,0.2,0.3], bins=10, title="Values")
  '''

  # Return early when data is None or empty.
  if (data is None or len(data) == 0):
    # Nothing to plot; return None.
    return None

  # Compute a default figure size when not provided.
  if (figSize is None):
    # Use a standard wide figure for histograms.
    figSize = (8.0, 4.0)

  # Create Figure and Axis for the histogram.
  fig, ax = plt.subplots(figsize=figSize)

  # Draw the histogram bars on the Axis.
  ax.hist(data, bins=bins, color="tab:blue", alpha=0.85)

  # Set x-axis label text.
  ax.set_xlabel(xlabel, fontsize=fontSize)

  # Set y-axis label text.
  ax.set_ylabel(ylabel, fontsize=fontSize)

  # Optionally set the title when provided.
  if (title and len(title) > 0):
    # Assign the title text to the Axes.
    ax.set_title(title, fontsize=fontSize)

  # Tighten layout to avoid clipping.
  fig.tight_layout()

  # Persist the figure when requested.
  if (save):
    # Determine the target path object for saving.
    targetPath: Optional[Path] = None
    # Prefer explicit savePath when provided.
    if (savePath is not None):
      # Convert savePath to a Path object.
      targetPath = Path(savePath)
    elif (fileName is not None):
      # Convert fileName to a Path object.
      targetPath = Path(fileName)
    # Save the figure when a target path exists.
    if (targetPath is not None):
      # Ensure parent directories exist for the target path.
      targetPath.parent.mkdir(parents=True, exist_ok=True)
      # Attempt to write the figure to the requested path.
      try:
        fig.savefig(str(targetPath), dpi=dpi, bbox_inches="tight")
      except Exception as e:
        # Print a helpful message on save failure.
        print(f"Error saving histogram: {e}")
      # Also attempt a PNG sibling when primary extension is not PNG.
      if (targetPath.suffix.lower().lstrip(".") != "png"):
        try:
          fig.savefig(str(targetPath.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
        except Exception:
          # Ignore PNG sibling save errors silently.
          pass

  # Legacy behavior: save when savePath provided even if save flag is False.
  if ((savePath is not None) and (not save) and (fileName is None)):
    # Attempt to persist to legacy savePath for compatibility.
    try:
      pathObj = Path(savePath)
      pathObj.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(str(pathObj), dpi=dpi, bbox_inches="tight")
      print(f"Histogram saved to: {pathObj}")
    except Exception:
      # Ignore legacy save errors silently.
      pass

  # Display the figure when requested.
  if (display):
    # Show the matplotlib figure.
    plt.show()

  # Close the figure to release memory.
  plt.close(fig)

  # Optionally return the created Figure object.
  if (returnFig):
    # Return the Figure instance.
    return fig

  # Default return when nothing to return.
  return None
