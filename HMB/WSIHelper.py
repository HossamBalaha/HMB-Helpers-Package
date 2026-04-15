# Import the required libraries.
import PIL, cv2, os, openslide, json, tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from HMB.Utils import *
from HMB.ImagesHelper import *


def ReadWSIViaOpenSlide(slidePath):
  r'''
  Reads a whole slide image (WSI) using OpenSlide and returns an OpenSlide object.

  Parameters:
    slidePath (str): Path to the slide file.

  Returns:
    openslide.OpenSlide: OpenSlide object for the slide.

  Raises:
    AssertionError: If the slide path does not exist.
  '''

  assert os.path.exists(slidePath), f"Slide path does not exist: {slidePath}"
  return openslide.OpenSlide(slidePath)


def ReadGeoJSONAnnotations(annotationFile):
  r'''
  Read a GeoJSON annotation file and return a list of annotations.

  Each annotation is a dict with the keys:
    - id: feature id (or index)
    - properties: feature properties (dict)
    - geometry: dict with 'type' and 'coordinates'

  Accepts either a pathlib.Path or a string path. Gracefully handles
  FeatureCollection, single Feature, or raw geometry objects.

  Parameters:
    annotationFile (str or pathlib.Path): Path to the GeoJSON annotation file.

  Returns:
    List[Dict[str, Any]]: List of annotations read from the GeoJSON file.
  '''

  p = Path(annotationFile)
  # Check that the provided file path exists.
  if (not p.exists()):
    print(f"ERROR: Annotation file does not exist: {p}")
    return []

  try:
    with p.open("r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception as e:
    # Report JSON parsing error with the path.
    print(f"ERROR: Failed to read/parse GeoJSON '{p}': {e}")
    return []

  annotations: List[Dict[str, Any]] = []

  # Normalize to a list of features.
  features = None
  if (isinstance(data, dict) and isinstance(data.get("features"), list)):
    features = data["features"]
  elif (isinstance(data, dict) and data.get("type") == "Feature"):
    features = [data]
  elif (isinstance(data, dict) and data.get("type") == "FeatureCollection"):
    features = data.get("features", [])
  else:
    # If data looks like a geometry object, wrap it as a single feature-like entry.
    if (isinstance(data, dict) and ("type" in data and "coordinates" in data)):
      features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
      print(f"WARNING: GeoJSON file '{p.name}' doesn't contain features or a geometry object; returning empty list")
      return []

  for idx, feat in enumerate(features):
    # Skip non-dictionary features.
    if (not isinstance(feat, dict)):
      print(f"WARNING: Skipping non-dict feature at index {idx}")
      continue

    fid = feat.get("id", idx)
    props = feat.get("properties", {}) or {}
    geom = feat.get("geometry") or feat  # sometimes the feature itself may be a geometry

    if (not isinstance(geom, dict) or "type" not in geom or "coordinates" not in geom):
      print(f"WARNING: feature {fid} missing valid geometry; skipping")
      continue

    gtype = geom.get("type")
    coords = geom.get("coordinates")

    # Append both the original GeoJSON-style keys and CamelCase aliases for compatibility.
    annotations.append(
      {
        "id"        : fid,
        "properties": props,
        "geometry"  : {"type": gtype, "coordinates": coords}
      }
    )

  print(f"Read {len(annotations)} annotations from: {p.name}")
  return annotations


def DrawPolygonOnImage(imageArr, coords, outlineColor=(255, 0, 0), fillColor=None, width=2):
  r'''
  Draw a polygon with optional filled semi-transparent interior.

  Parameters:
    imageArr (np.ndarray): Input image (HWC, uint8), RGB or BGR.
    coords (list): List of (x, y) tuples defining polygon vertices.
    outlineColor (tuple): RGB color for outline, e.g., (255, 0, 0) = red.
    fillColor (tuple or None): RGBA color for fill, e.g., (255, 0, 0, 0.5). If None, no fill.
    width (int): Thickness of the outline.

  Returns:
    numpy.ndarray: Modified image in same format as input (RGB assumed).
  '''

  # Work on a copy to avoid modifying original.
  image = np.array(imageArr).copy()

  # Handle grayscale: convert to 3-channel.
  if (image.ndim == 2):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  elif (image.shape[2] == 4):
    # Remove alpha channel (assume RGB(A) input).
    image = image[:, :, :3]

  # Ensure 3-channel BGR for OpenCV.
  if (image.shape[2] != 3):
    raise ValueError("Input image must be grayscale, RGB, or RGBA.")

  # Prepare points for OpenCV.
  pts = np.array([coords], dtype=np.int32)

  # Draw outline.
  cv2.polylines(
    image,
    pts,
    isClosed=True,
    color=outlineColor,
    thickness=width,
    lineType=cv2.LINE_AA
  )

  # Handle fill with transparency.
  if (fillColor is not None):
    if (len(fillColor) == 4):
      rgbFill = fillColor[:3]
      alpha = fillColor[3]
    else:
      rgbFill = fillColor
      alpha = 1.0

    # Create overlay for transparent fill.
    overlay = image.copy()
    cv2.fillPoly(overlay, pts, color=rgbFill)

    # Blend only if alpha < 1.0.
    if (alpha < 1.0):
      image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
    else:
      image = overlay

  return image


def ExtractPatchesFromWSI(
    wsi,
    wsiFile,
    annotations,
    outputDir,
    patchSize=(256, 256),
    overlap=(0, 0),
    maxNumPatchesPerAnnotation=100,
    label=None,
    tissueThreshold=0.3,  # Fraction of non-background pixels required
):
  r'''
  Extract patches from WSI within annotated regions, skipping background tiles.

  Parameters:
    wsi (openslide.OpenSlide): OpenSlide WSI object.
    wsiFile (str or Path): Path to the WSI file (for reference).
    annotations (list): List of GeoJSON-like annotation dicts.
    outputDir (str or Path): Directory to save extracted patches.
    patchSize (tuple): Size of patches to extract (width, height).
    overlap (tuple): Overlap between patches (x_overlap, y_overlap).
    maxNumPatchesPerAnnotation (int): Max patches to extract per annotation.
    label: Optional label to embed in patch filenames.
    tissueThreshold (float): Minimum fraction of tissue pixels required.
  '''

  import cv2
  from PIL import Image
  from shapely.geometry import Point, Polygon

  # Create the output path if missing.
  outputPath = Path(outputDir)
  outputPath.mkdir(parents=True, exist_ok=True)

  # Validate patch size and overlap.
  if (patchSize[0] <= 0 or patchSize[1] <= 0):
    raise ValueError("Patch size must be positive.")
  if (overlap[0] >= patchSize[0] or overlap[1] >= patchSize[1]):
    raise ValueError("Overlap must be less than patch size.")

  for annotIdx, annot in enumerate(annotations):
    coordsList = np.array(annot["geometry"]["coordinates"][0])
    polygon = Polygon(coordsList)

    xCoords = coordsList[:, 0]
    yCoords = coordsList[:, 1]
    minX, maxX = int(np.floor(np.min(xCoords))), int(np.ceil(np.max(xCoords)))
    minY, maxY = int(np.floor(np.min(yCoords))), int(np.ceil(np.max(yCoords)))

    patchCount = 0
    stepX = max(1, patchSize[0] - overlap[0])
    stepY = max(1, patchSize[1] - overlap[1])

    for y in range(minY, maxY - patchSize[1] + 1, stepY):
      for x in range(minX, maxX - patchSize[0] + 1, stepX):
        if patchCount >= maxNumPatchesPerAnnotation:
          break

        # Skip if patch center is outside annotation.
        centerX = x + patchSize[0] // 2
        centerY = y + patchSize[1] // 2
        if not polygon.contains(Point(centerX, centerY)):
          continue

        try:
          # Read patch region.
          patch = wsi.read_region((x, y), 0, patchSize)
          if (patch.mode != "RGB"):
            patch = patch.convert("RGB")
          patchArr = np.array(patch)  # Shape: (H, W, 3).

          # Background filtering.
          # Strategy: skip if too many white (>240) or black (<15) pixels.
          # Common in WSI: background = white; artifacts = black.
          whiteMask = (patchArr[:, :, 0] > 240) & \
                      (patchArr[:, :, 1] > 240) & \
                      (patchArr[:, :, 2] > 240)
          blackMask = (patchArr[:, :, 0] < 15) & \
                      (patchArr[:, :, 1] < 15) & \
                      (patchArr[:, :, 2] < 15)
          backgroundMask = whiteMask | blackMask
          backgroundFraction = np.mean(backgroundMask)

          if (backgroundFraction > (1.0 - tissueThreshold)):
            continue  # Skip background-dominant patch

          # Save patch with embedded label and coordinates.
          safeLabel = (
            str(label).replace("/", "_").replace("\\", "_")
            if (label is not None) else "Unlabeled"
          )
          wsiFileName = str(Path(wsiFile).stem)
          patchFilename = outputPath / f"{wsiFileName}_A{annotIdx + 1}_X{x}_Y{y}_L{safeLabel}.jpg"

          # Use PIL or cv2 to save (more reliable than plt.imsave for JPG).
          imgPIL = Image.fromarray(patchArr)
          # Save with maximum quality.
          imgPIL.save(patchFilename, quality=100)
          patchCount += 1

        except Exception as e:
          print(f"Warning: Failed to process patch at ({x}, {y}): {e}")
          continue

      if (patchCount >= maxNumPatchesPerAnnotation):
        break

    print(f"Extracted {patchCount} valid tissue patches for annotation {annotIdx + 1}")


def TileExtractionAlignmentHandler(
    heSlidePath,  # Path to the HE slide.
    mtSlidePath,  # Path to the MT slide.
    storageDir,  # Directory to store extracted patches and visualizations.
    patchesPerSlide=5000,  # Total number of patches to extract per slide.
    targetSize=(512, 512),  # Size of each patch to extract.
    regionSize=(256 * 16, 256 * 16),  # Size of the region used for processing.
    overlapSize=(256, 256),  # Overlap size between adjacent patches.
    thumbnailSize=(1024, 1024),  # Size of the thumbnail for visualization.
    toleranceSIFT=0.50,  # Tolerance for SIFT matching (default is 0.50).
    maxNumFeaturesORB=5000,  # Maximum number of features to detect for matching.
    maxGoodMatchesORB=25,  # Maximum number of good matches to consider for alignment.
    emptyPercentageThreshold=80,  # Threshold for empty percentage in patches (default is 80%).
    doPlotting=True,  # Whether to generate and save plots for visualization.
    verbose=False,  # Whether to print verbose output during processing.
    dpi=720,  # DPI for saved plots. Default is 720 for high-quality visualization.
):
  r'''
  Extracts and aligns patches from HE (Hematoxylin-Eosin) and MT (Masson's Trichrome) slides,
  processes them, and saves the results.

  Parameters:
    heSlidePath (str): Path to the HE slide.
    mtSlidePath (str): Path to the MT slide.
    storageDir (str): Directory to store extracted patches and visualizations.
    patchesPerSlide (int): Total number of patches to extract per slide.
    targetSize (tuple): Size of each patch to extract (width, height).
    regionSize (tuple): Size of the region used for processing.
    overlapSize (tuple): Overlap between adjacent patches.
    thumbnailSize (tuple): Size of the thumbnail for visualization.
    toleranceSIFT (float): Tolerance for SIFT matching (default is 0.50).
    maxNumFeaturesORB (int): Maximum number of features to detect for matching.
    maxGoodMatchesORB (int): Maximum number of good matches to consider for alignment.
    emptyPercentageThreshold (int): Threshold for empty percentage in patches (default is 80%).
    doPlotting (bool): Whether to generate and save plots for visualization.
    verbose (bool): Whether to print verbose output during processing.
    dpi (int): DPI for saved plots.

  Raises:
    AssertionError: If the HE or MT slide paths do not exist.
  '''

  # Ensure the HE and MT slide paths exist. If not, raise an assertion error with a descriptive message.
  assert os.path.exists(heSlidePath), f"HE path does not exist: {heSlidePath}"
  assert os.path.exists(mtSlidePath), f"MT path does not exist: {mtSlidePath}"

  # Define the storage paths for HE and MT tiles.
  heTilesStoragePath = os.path.join(storageDir, "HE")
  mtTilesStoragePath = os.path.join(storageDir, "MT")
  # Create the directories if they do not exist.
  os.makedirs(heTilesStoragePath, exist_ok=True)
  os.makedirs(mtTilesStoragePath, exist_ok=True)

  # If doPlotting is True, create a directory for storing thumbnails.
  if (doPlotting):
    thumbStoragePath = os.path.join(storageDir, "Thumbnails Visualization")
    os.makedirs(thumbStoragePath, exist_ok=True)

  # Open the HE and MT slides using the OpenSlide library.
  # This allows access to high-resolution slide data and metadata.
  heSlide = openslide.OpenSlide(heSlidePath)
  mtSlide = openslide.OpenSlide(mtSlidePath)

  # Generate thumbnails for the HE and MT slides at the specified size.
  # Thumbnails provide a low-resolution overview of the slides for visualization and alignment.
  heThumb = heSlide.get_thumbnail(thumbnailSize)
  mtThumb = mtSlide.get_thumbnail(thumbnailSize)
  # Convert the thumbnails to NumPy arrays for further processing.
  heThumb = np.array(heThumb)
  mtThumb = np.array(mtThumb)

  # Extract contours from the HE and MT thumbnails.
  # Contours represent the boundaries of regions of interest (e.g., tissue areas).
  heThumb, heContour, heMask, heDraw = ExtractLargestContour(heThumb.copy())
  mtThumb, mtContour, mtMask, mtDraw = ExtractLargestContour(mtThumb.copy())

  # Match the HE and MT thumbnails using SIFT (Scale-Invariant Feature Transform).
  # SIFT identifies keypoints and matches them between the two images to compute a homography matrix.
  heThumb, mtThumb, matched, thumbnailHomography, shape = MatchTwoImagesViaSIFT(
    heThumb.copy(), mtThumb.copy(), shape=None, tolerance=toleranceSIFT,
  )

  # Plot the results if doPlotting is True.
  if (doPlotting):
    fig = plt.figure()
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=1)
    ax1.imshow(heDraw)
    ax1.set_title("HE w/ Contour")
    ax1.axis("off")
    ax2 = plt.subplot2grid((2, 4), (0, 1), colspan=1)
    ax2.imshow(mtDraw)
    ax2.set_title("MT w/ Contour")
    ax2.axis("off")
    ax3 = plt.subplot2grid((2, 4), (0, 2), colspan=1)
    ax3.imshow(heThumb)
    ax3.set_title("HE Transformed")
    ax3.axis("off")
    ax4 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
    ax4.imshow(mtThumb)
    ax4.set_title("MT Transformed")
    ax4.axis("off")
    ax5 = plt.subplot2grid((2, 4), (1, 0), colspan=4)
    ax5.imshow(matched)
    ax5.set_title("Matched")
    ax5.axis("off")
    plt.tight_layout()
    # Save the plot as PNG only if thumbStoragePath is defined.
    plt.savefig(
      os.path.join(storageDir, "Thumbnails Visualization", f"Slide_{os.path.basename(heSlidePath)}.png"),
      dpi=dpi,
      bbox_inches="tight",
    )
    plt.close()  # Close the plot to free up memory.

  # Calculate the bounding box of the HE contour.
  # The bounding box defines the rectangular region enclosing the contour.
  heContourBox = cv2.boundingRect(heContour)
  # Calculate the maximum downsampling factor for the HE slide.
  maxSampling = int(heSlide.level_downsamples[heSlide.level_count - 1])
  # Calculate the center point of the HE contour.
  centerPoint = (
    (heContourBox[0] + heContourBox[2] // 2) * maxSampling,
    (heContourBox[1] + heContourBox[3] // 2) * maxSampling,
  )
  # Calculate the number of patches per side based on the total number of patches.
  patchesPerSide = int(np.sqrt(patchesPerSlide) / 2.0)
  # Loop through each patch location around the center point.
  for i in range(-patchesPerSide, patchesPerSide + 1):
    for j in range(-patchesPerSide, patchesPerSide + 1):
      try:
        # Calculate the top-left corner of the patch.
        topLeftPoint = (
          centerPoint[0] + i * (targetSize[0] - overlapSize[0]),
          centerPoint[1] + j * (targetSize[1] - overlapSize[1]),
        )

        # Define file paths for saving the HE and MT patches.
        postfix = f"{os.path.basename(heSlidePath)}_{topLeftPoint}"
        heRegionOrigPath = os.path.join(heTilesStoragePath, f"HE_{postfix}.png")
        mtRegionOrigPath = os.path.join(mtTilesStoragePath, f"MT_{postfix}.png")

        # Skip the patch if it already exists.
        if (os.path.exists(heRegionOrigPath) and os.path.exists(mtRegionOrigPath)):
          if (verbose):  # Print a message if verbose mode is enabled.
            print(f"[SKIPPED] Patch already exists.")
          continue

        # Extract the patch from the HE and MT slides using the computed point and homography.
        # The `ExtractPatch` function handles alignment and extraction of the corresponding regions.
        (
          heRegionOrig, mtRegionOrig,
          heRegionBW, mtRegionBW, diff,
          weighted, flag,
        ) = ExtractPatch(
          topLeftPoint,  # Top-left corner of the patch to extract.
          heSlide,  # HE slide object.
          mtSlide,  # MT slide object.
          mtContour,  # Contour defining the boundaries of the MT slide region.
          heContour,  # Contour defining the boundaries of the HE slide region.
          regionSize,  # Size of the region to extract (width, height).
          targetSize,  # Target size for the final extracted patch (width, height).
          thumbnailHomography,  # Homography matrix for transforming coordinates.
          maxNumFeaturesORB=maxNumFeaturesORB,  # Maximum number of features to detect for matching.
          maxGoodMatchesORB=maxGoodMatchesORB,  # Maximum number of good matches to consider for alignment.
        )

        # Print the patch location and flag status.
        if (verbose):  # Print a message if verbose mode is enabled.
          print(f"{os.path.basename(heSlidePath)} => Top Left Point ({topLeftPoint}): {flag}.")

        # Skip the patch if the flag indicates failure.
        if (not flag):
          continue

        # Calculate the empty percentage of the HE and MT patches.
        # Empty percentage measures the proportion of non-tissue pixels in the patch.
        hePathEmptyPercentage = GetEmptyPercentage(heRegionOrig, targetSize)
        mtPathEmptyPercentage = GetEmptyPercentage(mtRegionOrig, targetSize)
        # Round the empty percentages to 2 decimal places for readability.
        hePathEmptyPercentage = np.round(hePathEmptyPercentage, 2)
        mtPathEmptyPercentage = np.round(mtPathEmptyPercentage, 2)

        # Skip the patch if the empty percentage exceeds the threshold.
        if ((hePathEmptyPercentage > emptyPercentageThreshold) or (mtPathEmptyPercentage > emptyPercentageThreshold)):
          if (verbose):  # Print a message if verbose mode is enabled.
            print(f"[SKIPPED] Empty Percentage: {hePathEmptyPercentage}, {mtPathEmptyPercentage}")
          continue

        # Check if the similarity between the HE and MT patches meets the acceptance criteria.
        isOK, reason = IsSimilarityAccepted(heRegionOrig, mtRegionOrig)
        if (not isOK):
          if (verbose):  # Print a message if verbose mode is enabled.
            print(f"[SKIPPED] Similarity not accepted.")
            print(f"Reason: {reason}")
          continue

        # Save the HE and MT patches as PNG files.
        heRegionOrig.save(heRegionOrigPath)
        mtRegionOrig.save(mtRegionOrigPath)

        # # Convert the HE and MT regions to NumPy arrays for further analysis.
        # heRegion = np.array(heRegionOrig)
        # mtRegion = np.array(mtRegionOrig)

        # # Optionally plot the HE and MT regions and their processed versions if doPlotting is True.
        # if (doPlotting):
        #   plt.figure()
        #   plt.subplot(2, 3, 1)
        #   plt.imshow(heRegion)
        #   plt.tight_layout()
        #   plt.axis("off")
        #   plt.subplot(2, 3, 2)
        #   plt.imshow(mtRegion)
        #   plt.tight_layout()
        #   plt.axis("off")
        #   plt.subplot(2, 3, 3)
        #   plt.imshow(weighted, cmap="gray")
        #   plt.tight_layout()
        #   plt.axis("off")
        #   plt.subplot(2, 3, 4)
        #   plt.imshow(heRegionBW, cmap="gray")
        #   plt.tight_layout()
        #   plt.axis("off")
        #   plt.subplot(2, 3, 5)
        #   plt.imshow(mtRegionBW, cmap="gray")
        #   plt.tight_layout()
        #   plt.axis("off")
        #   plt.subplot(2, 3, 6)
        #   plt.imshow(diff, cmap="gray")
        #   plt.tight_layout()
        #   plt.axis("off")
        #   # Save the plot as PNG.
        #   plt.savefig(
        #     os.path.join(mtTilesStoragePath, f"Tile_{topLeftPoint}.png"),
        #     dpi=dpi,
        #     bbox_inches="tight",
        #   )
        #   plt.close()

      except Exception as e:
        # Handle any exceptions that occur during patch extraction or processing.
        if (verbose):  # Print a message if verbose mode is enabled.
          print(f"Error: {e}")
        continue


def ExtractPatch(
    topLeftRegion,  # Top-left corner coordinates of the region to extract.
    heSlide,  # HE slide object.
    mtSlide,  # MT slide object.
    mtContour,  # Contour defining the boundaries of the MT slide region.
    heContour,  # Contour defining the boundaries of the HE slide region.
    regionSize,  # Size of the region to extract (width, height).
    targetSize,  # Target size for the final extracted patch (width, height).
    homography,  # Homography matrix for transforming coordinates.
    maxNumFeaturesORB=5000,  # Maximum number of features to detect for matching.
    maxGoodMatchesORB=25,  # Maximum number of good matches to consider for alignment.
):
  r'''
  Extracts and processes patches from HE (Hematoxylin & Eosin) and MT (Trichrome) slides.
  The function applies transformations, matches regions using ORB, and computes differences
  between the two slide types. It also generates binary masks and weighted combinations.

  Parameters:
    topLeftRegion (tuple): Top-left corner coordinates of the region to extract.
    heSlide (openslide.OpenSlide): HE slide object.
    mtSlide (openslide.OpenSlide): MT slide object.
    mtContour (list): Contour defining the boundaries of the MT slide region.
    heContour (list): Contour defining the boundaries of the HE slide region.
    regionSize (tuple): Size of the region to extract (width, height).
    targetSize (tuple): Target size for the final extracted patch (width, height).
    homography (numpy.ndarray): Homography matrix for transforming coordinates.
    maxNumFeaturesORB (int): Maximum number of features to detect for matching.
    maxGoodMatchesORB (int): Maximum number of good matches to consider for alignment.

  Returns:
    tuple: A tuple containing:
      - heRegionOrig (PIL.Image): Original HE region at the target size.
      - mtRegionOrig (PIL.Image): Original MT region at the target size.
      - heRegionBW (numpy.ndarray): Binary mask of the HE region.
      - mtRegionBW (numpy.ndarray): Binary mask of the MT region.
      - diff (numpy.ndarray): Absolute difference between the binary masks.
      - weighted (numpy.ndarray): Weighted combination of the HE and MT regions.
      - topLeftFlag (bool): Flag indicating if the top-left point is inside both contours.
  '''

  # Calculate the maximum downsampling factor for the HE slide.
  maxSampling = int(heSlide.level_downsamples[heSlide.level_count - 1])

  # Downsample the top-left region coordinates based on the maximum downsampling factor.
  topLeftDownsampled = [
    int(np.round(topLeftRegion[0] / maxSampling)),
    int(np.round(topLeftRegion[1] / maxSampling))
  ]

  # Convert the downsampled coordinates to homogeneous coordinates for transformation.
  topLeftDownHomogeneous = np.array([topLeftDownsampled[0], topLeftDownsampled[1], 1])

  # Check if the downsampled point lies within both the MT and HE contours.
  topLeftFlag = (
      IsPointInsideContour(topLeftDownsampled, mtContour) and
      IsPointInsideContour(topLeftDownsampled, heContour)
  )

  # Apply the homography transformation to the downsampled point.
  topLeftWrapped = homography @ topLeftDownHomogeneous

  # Normalize the homogeneous coordinates after transformation.
  topLeftWrapped = topLeftWrapped[:2] / topLeftWrapped[2]

  # Scale the wrapped coordinates back to the original resolution.
  topLeftWrapped = [
    int(np.round(topLeftWrapped[0] * maxSampling)),
    int(np.round(topLeftWrapped[1] * maxSampling))
  ]

  # Extract the MT region from the MT slide at the transformed coordinates.
  mtRegion = mtSlide.read_region(
    topLeftWrapped,  # Top-left corner of the region.
    0,  # Level (0 is the highest resolution).
    regionSize,  # Size of the region.
  )

  # Convert the MT region to a NumPy array for further processing.
  mtRegion = np.array(mtRegion)

  # Extract the HE region from the HE slide at the transformed coordinates.
  heRegion = heSlide.read_region(
    topLeftWrapped,  # Top-left corner of the region.
    0,  # Level (0 is the highest resolution).
    regionSize,  # Size of the region.
  )

  # Convert the HE region to a NumPy array for further processing.
  heRegion = np.array(heRegion)

  # Match the HE and MT regions using ORB feature matching.
  heRegion, mtRegion, tileMatched, tileHomography, tileShape = MatchTwoImagesViaORB(
    heRegion.copy(),  # HE region to match.
    mtRegion.copy(),  # MT region to match.
    shape=None,  # Use the maximum dimensions of the input images.
    maxNumFeatures=maxNumFeaturesORB,  # Maximum number of features to detect for matching.
    maxGoodMatches=maxGoodMatchesORB,  # Maximum number of good matches to consider for alignment.
  )

  # Calculate the center of the matched region.
  centerBlock = (tileShape[0] // 2, tileShape[1] // 2)

  # Calculate the top-left corner of the target region based on the center.
  topLeft = (
    centerBlock[0] - targetSize[0] // 2,
    centerBlock[1] - targetSize[1] // 2,
  )

  # Convert the top-left coordinates to homogeneous coordinates for transformation.
  topLeftHomogeneous = np.array([topLeft[0], topLeft[1], 1])

  # Apply the homography transformation to the top-left coordinates.
  pointWrapped = tileHomography @ topLeftHomogeneous

  # Normalize the homogeneous coordinates after transformation.
  pointWrapped = pointWrapped[:2] / pointWrapped[2]

  # Scale the wrapped coordinates back to the original resolution.
  pointWrapped = (
    int(np.round(pointWrapped[0]) + topLeftWrapped[0]),
    int(np.round(pointWrapped[1]) + topLeftWrapped[1]),
  )

  # Calculate the non-wrapped coordinates for the HE region.
  pointNotWrapped = (
    int(np.round(topLeft[0] + topLeftWrapped[0])),
    int(np.round(topLeft[1] + topLeftWrapped[1])),
  )

  # Extract the HE region at the target size.
  heRegionOrig = heSlide.read_region(pointNotWrapped, 0, targetSize)

  # Extract the MT region at the target size.
  mtRegionOrig = mtSlide.read_region(pointWrapped, 0, targetSize)

  # Convert the HE and MT regions to NumPy arrays.
  heRegion = np.array(heRegionOrig)
  mtRegion = np.array(mtRegionOrig)

  # Convert the HE and MT regions to grayscale for binary mask generation.
  heRegionBW = cv2.cvtColor(heRegion, cv2.COLOR_RGB2GRAY)
  mtRegionBW = cv2.cvtColor(mtRegion, cv2.COLOR_RGB2GRAY)

  # Threshold the grayscale images to create binary masks.
  heRegionBW = cv2.threshold(heRegionBW, 225, 255, cv2.THRESH_BINARY_INV)[1]
  mtRegionBW = cv2.threshold(mtRegionBW, 225, 255, cv2.THRESH_BINARY_INV)[1]

  # Calculate the absolute difference between the binary masks.
  diff = cv2.absdiff(heRegionBW, mtRegionBW)

  # Create a weighted combination of the HE and MT regions for visualization.
  weighted = cv2.addWeighted(heRegion, 0.5, mtRegion, 0.5, 0.0)

  # Return the extracted regions and their processed versions.
  return (
    heRegionOrig, mtRegionOrig,
    heRegionBW, mtRegionBW, diff,
    weighted, topLeftFlag,
  )


def FreeFormDeformationHandler(
    heFolderPath,  # Path to the folder containing HE images.
    mtFolderPath,  # Path to the folder containing corresponding MT images.
    heDeformedFolderPath,  # Path to the folder where deformed HE images will be saved.
    doPlotting=False,  # Whether to store visualizations.
    visualizationFolderPath=None,  # Path to the folder for storing visualizations (optional, default is None).
    gridSize=[10, 10],  # Grid size for the B-spline transform.
    numberOfHistogramBins=50,  # Number of histogram bins for the metric.
    samplingPercentage=0.1,  # Percentage of pixels to sample for the metric.
    learningRate=0.01,  # Learning rate for the optimizer.
    numberOfIterations=500,  # Maximum number of iterations.
    convergenceMinimumValue=1e-6,  # Convergence threshold.
    convergenceWindowSize=10,  # Window size for convergence determination.
    verbose=False,  # Whether to print verbose output during processing.
):
  r'''
  Apply Free Form Deformation (FFD) to HE images and save the deformed results.
  This function applies B-spline based Free Form Deformation to each HE image, 
  optimizing the deformation to align with the corresponding MT image. 
  The optimization is performed using a metric computed from sampled pixels and histogram bins, 
  and the process is controlled by a learning rate and convergence criteria.

  .. math::
    \text{FFD}(x, y) = \sum_{i=0}^{n} \sum_{j=0}^{m} B_i(u) \cdot B_j(v) \cdot P_{i,j}

  where
    - :math:`B_i(u)` and :math:`B_j(v)` are B-spline basis functions.
    - :math:`P_{i,j}` are control points.

  Parameters:
    heFolderPath (str): Path to the folder containing HE images.
    mtFolderPath (str): Path to the folder containing corresponding MT images.
    heDeformedFolderPath (str): Path to the folder where deformed HE images will be saved.
    doPlotting (bool): Whether to store visualizations (default is False).
    visualizationFolderPath (str): Path to the folder for storing visualizations (optional).
    gridSize (list): Grid size for the B-spline transform (default is [10, 10]).
    numberOfHistogramBins (int): Number of histogram bins for the metric (default is 50).
    samplingPercentage (float): Percentage of pixels to sample for the metric (default is 0.1).
    learningRate (float): Learning rate for the optimizer (default is 0.01).
    numberOfIterations (int): Maximum number of iterations (default is 500).
    convergenceMinimumValue (float): Convergence threshold (default is 1e-6).
    convergenceWindowSize (int): Window size for convergence determination (default is 10).
    verbose (bool): Whether to print verbose output during processing (default is False).

  Raises:
    AssertionError: If the HE or MT folder paths do not exist.
  '''

  # Ensure the HE folder path exists. If not, raise an assertion error with a descriptive message.
  assert os.path.exists(heFolderPath), f"HE folder path does not exist: {heFolderPath}"
  # Ensure the MT folder path exists. If not, raise an assertion error with a descriptive message.
  assert os.path.exists(mtFolderPath), f"MT folder path does not exist: {mtFolderPath}"

  # List all files in the HE directory to process each image.
  heFiles = os.listdir(heFolderPath)  # Retrieve all filenames in the HE folder.

  # Create the directory for storing deformed images if it does not already exist.
  os.makedirs(heDeformedFolderPath, exist_ok=True)  # Ensure the output directory exists.

  # Loop through each HE file in the directory using a progress bar for feedback.
  for heFile in tqdm.tqdm(heFiles):  # Use tqdm to display progress during processing.
    try:
      # Define the full path to the HE image file.
      imagePath1 = os.path.join(heFolderPath, heFile)  # Construct the path to the HE image.

      # Check if the HE image exists.
      if (not os.path.exists(imagePath1)):  # If the HE image does not exist, skip this file.
        if (verbose):  # Print a message if verbose mode is enabled.
          print(f"[SKIPPED] HE image not found: {heFile}.")
        continue  # Skip to the next HE file if the image is missing.

      # Define the full path to the corresponding MT image file.
      # Replace "HE_" with "MT_" in the filename to match the MT image naming convention.
      imagePath2 = os.path.join(mtFolderPath, heFile.replace("HE_", "MT_"))  # Construct the path to the MT image.

      # Check if the corresponding MT image exists.
      if (not os.path.exists(imagePath2)):  # If the MT image does not exist, skip this HE image.
        if (verbose):  # Print a message if verbose mode is enabled.
          print(f"[SKIPPED] Corresponding MT image not found for {heFile}.")
        continue  # Skip to the next HE file if the MT image is missing.

      # Define the path where the deformed HE image will be saved.
      storePath = os.path.join(heDeformedFolderPath, heFile)  # Construct the path for the deformed image.

      # Skip processing if the deformed image already exists at the specified path.
      if (os.path.exists(storePath)):  # Check if the file already exists.
        continue  # Skip to the next file if the deformed image is already saved.

      # Apply Free Form Deformation (FFD) to align the HE image with the MT image.
      image1, image2, deformed, deformationField = FreeFormDeformationImproved(
        imagePath1, imagePath2,
        gridSize=gridSize,  # Grid size for the B-spline transform.
        numberOfHistogramBins=numberOfHistogramBins,  # Number of histogram bins for the metric.
        samplingPercentage=samplingPercentage,  # Percentage of pixels to sample for the metric.
        learningRate=learningRate,  # Learning rate for the optimizer.
        numberOfIterations=numberOfIterations,  # Maximum number of iterations.
        convergenceMinimumValue=convergenceMinimumValue,  # Convergence threshold.
        convergenceWindowSize=convergenceWindowSize,  # Window size for convergence determination.
      )

      # Convert the deformed NumPy array back into a PIL image for saving.
      deformedImage = PIL.Image.fromarray(deformed)  # Convert the NumPy array to a PIL image.

      # Copy metadata from the original HE image to the deformed image.
      deformedImage.info = image1.info  # Preserve metadata such as resolution and format.

      # Save the deformed image to the specified path.
      deformedImage.save(storePath)  # Save the deformed image to disk.

      # Optionally store visualizations if requested.
      if (doPlotting and (visualizationFolderPath is not None)):
        # Ensure the visualization directory exists.
        os.makedirs(visualizationFolderPath, exist_ok=True)  # Create the visualization folder if needed.

        # Compute the gradient of the deformation field.
        dx, dy = np.gradient(deformationField, axis=(0, 1))  # Compute gradients along x and y directions.
        gradientMagnitude = np.sqrt(dx ** 2 + dy ** 2)  # Compute the magnitude of the gradient.

        # Plot the deformation field.
        plt.figure()
        plt.streamplot(
          np.arange(deformationField[::5, ::5, 0].shape[1]),  # X-coordinates for the streamplot.
          np.arange(deformationField[::5, ::5, 1].shape[0]),  # Y-coordinates for the streamplot.
          deformationField[::5, ::5, 0],  # X-component of the deformation field.
          deformationField[::5, ::5, 1],  # Y-component of the deformation field.
          color=gradientMagnitude[::5, ::5, 0],  # Color based on the gradient magnitude.
          cmap="viridis",  # Colormap for visualization.
          arrowsize=1.5,  # Size of the arrows in the streamplot.
        )
        plt.colorbar(label="Gradient Magnitude")  # Add a colorbar to indicate gradient magnitude.
        plt.xlabel("X-axis")  # Label for the x-axis.
        plt.ylabel("Y-axis")  # Label for the y-axis.
        plt.axis("off")
        plt.title("Deformation Field using Streamplot")  # Title for the plot.
        plt.tight_layout()
        # Save the deformation field plot.
        deformationFieldPath = os.path.join(visualizationFolderPath, heFile.replace(".png", "_DeformationField.png"))
        plt.savefig(deformationFieldPath, bbox_inches="tight", dpi=720)  # Save the plot without padding.
        plt.close()  # Close the plot to free up memory.

        # Define the path for the visualization file.
        visualizationPath = os.path.join(visualizationFolderPath, heFile)  # Construct the visualization path.
        overlayImage = cv2.addWeighted(
          np.array(image1),  # Convert the original HE image to BGR for OpenCV.
          0.5,  # Weight for the original HE image.
          np.array(deformedImage),  # Convert the deformed image to BGR.
          0.5,  # Weight for the deformed image.
          0,  # No additional scalar added to the sum.
        )
        overlayImage = PIL.Image.fromarray(overlayImage)  # Convert the overlay image back to PIL format.
        overlayImage.info = image1.info  # Copy metadata from the original HE image.
        # Calculate the absolute difference between the original and deformed images.
        subImage = np.abs(np.array(image1) - np.array(deformedImage))
        subImage = PIL.Image.fromarray(subImage).convert("L")  # Convert the difference image to grayscale.
        # Save the visualization (details depend on implementation).
        # Example: Save side-by-side comparison of original and deformed images.
        newPILImage = PIL.Image.new("RGBA", (image1.width * 5 + 50 * 4, image1.height))
        # Fill with a transparent background.
        newPILImage.paste((0, 0, 0, 0), (0, 0, newPILImage.width, newPILImage.height))  # Transparent background.
        newPILImage.paste(image1, (0, 0))  # Paste the original HE image.
        newPILImage.paste(image2, (image2.width + 50, 0))  # Paste the deformed image next to it.
        newPILImage.paste(deformedImage, (image2.width * 2 + 50 * 2, 0))  # Paste the deformed image.
        newPILImage.paste(overlayImage, (image2.width * 3 + 50 * 3, 0))  # Paste the overlay image.
        newPILImage.paste(subImage, (image2.width * 4 + 50 * 4, 0))
        newPILImage.info = image1.info  # Copy metadata from the original HE image.
        # Save the visualization image.
        newPILImage.save(visualizationPath)  # Save the visualization image.

    except Exception as e:
      # Handle any exceptions that occur during processing.
      print(f"Error processing {heFile}: {e}")


def ExtractRandomTilesFromImages(
    labelsFile,  # Path to CSV labels or a pandas.DataFrame.
    slidesDir="Slides",  # Directory where slide files live (joined with filenames from labels).
    outputDir="Tiles",  # Root output directory where class folders will be created.
    targetShape=(256, 256),  # (width, height) of tiles to extract.
    numOfTiles=1000,  # Number of tiles to extract per slide.
    allowedBackgroundRatio=0.65,  # Maximum allowed ratio of (mostly) background pixels.
    filenameColumn="filename",  # Column name in labels CSV that contains filenames.
    categoryColumn="Category",  # Column name that contains category / class integer.
    maxAttemptsFactor=10,  # Maximum attempts = numOfTiles * maxAttemptsFactor (avoid infinite loops).
    verbose=False,  # Verbose logging.
):
  r'''
  Extract random tiles from a set of large images using pyvips for efficient IO and
  OpenCV for simple background filtering. This mirrors the user's provided script but
  is wrapped as a reusable function following the repository style.
  '''

  # Local imports (keep module-level imports unchanged).
  import pyvips, math
  import pandas as pd

  # Load labels.
  if (isinstance(labelsFile, (str, Path))):
    if (not os.path.exists(str(labelsFile))):
      raise FileNotFoundError(f"Labels file not found: {labelsFile}")
    labels = pd.read_csv(labelsFile)
  elif (hasattr(labelsFile, "copy") and isinstance(labelsFile, pd.DataFrame)):
    labels = labelsFile.copy()
  else:
    raise ValueError("labelsFile must be a path to a CSV or a pandas.DataFrame")

  # If category column missing, try to build it from one-hot columns (common pattern).
  if (categoryColumn not in labels.columns):
    # assume first column is filename and rest are one-hot class columns.
    if (labels.shape[1] >= 2):
      catCols = labels[labels.columns[1:]].columns
      try:
        labels["Category"] = np.argmax(labels[catCols].values, axis=1)
        categoryColumn = "Category"
        if (verbose):
          print("Inferred 'Category' from one-hot columns.")
      except Exception:
        raise ValueError(f"Cannot infer a '{categoryColumn}' column from labels; provide explicit column name.")
    else:
      raise ValueError(f"labels DataFrame doesn't contain a '{categoryColumn}' column and cannot infer one.")

  results = {}

  # Ensure output base exists.
  os.makedirs(outputDir, exist_ok=True)

  for idx, row in labels.iterrows():
    filename = str(row[filenameColumn])
    cat = row[categoryColumn]
    clsPath = os.path.join(outputDir, str(cat))
    os.makedirs(clsPath, exist_ok=True)

    filePath = os.path.join(slidesDir, filename)
    # Check slide existence.
    if (not os.path.exists(filePath)):
      if (verbose):
        print(f"[SKIP] slide not found: {filePath}")
      results[filename] = 0
      continue

    try:
      image = pyvips.Image.new_from_file(filePath, access="sequential")
    except Exception as e:
      if (verbose):
        print(f"[ERROR] failed to open {filePath}: {e}")
      results[filename] = 0
      continue

    width, height = image.width, image.height
    tw, th = int(targetShape[0]), int(targetShape[1])

    if (width <= tw or height <= th):
      if (verbose):
        print(f"[SKIP] slide smaller than target tile size: {filePath} ({width}x{height})")
      results[filename] = 0
      continue

    counter = 0
    attempts = 0
    maxAttempts = max(1000, int(numOfTiles * maxAttemptsFactor))

    # Loop until requested tiles are obtained or attempts exhausted.
    while (counter < numOfTiles and attempts < maxAttempts):
      attempts += 1
      x = np.random.randint(0, width - tw)
      y = np.random.randint(0, height - th)

      try:
        tileVips = image.crop(x, y, tw, th)
        tile = np.ndarray(
          buffer=tileVips.write_to_memory(),
          dtype=np.uint8,
          shape=[tileVips.height, tileVips.width, 3]
        )

        # Convert to HSV then to gray as in the original snippet.
        imgHSV = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        imgGray = cv2.cvtColor(imgHSV, cv2.COLOR_BGR2GRAY)
        imgGray = cv2.GaussianBlur(imgGray, (3, 3), 0)
        imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        ratioOfBlackBackground = np.sum(imgThresh <= 255 * 0.1) / float(tw * th)
        ratioOfWhiteBackground = np.sum(imgThresh >= 255 * 0.9) / float(tw * th)

        if ((ratioOfBlackBackground > allowedBackgroundRatio) or (ratioOfWhiteBackground > allowedBackgroundRatio)):
          continue

        # Save tile.
        baseName = os.path.splitext(os.path.basename(filename))[0]
        outName = os.path.join(clsPath, f"{baseName}_{x}_{y}_{counter}.jpg")
        # cv2.imwrite expects BGR; tile is assumed compatible with user's original snippet.
        cv2.imwrite(outName, tile)
        counter += 1

      except Exception as e:
        if (verbose):
          print(f"[WARN] failed cropping/saving tile from {filePath} at ({x},{y}): {e}")
        continue

    results[filename] = counter
    if (verbose):
      print(f"Saved {counter}/{numOfTiles} tiles for {filename} (attempts: {attempts})")

  return results


def ExtractBACHAnnotationsFromXML(xmlFile, verbose=True):
  r'''
  Extract annotations from a BACH XML file.

  Parameters:
    xmlFile (str): Path to the XML file containing annotations.
    verbose (bool): Whether to print debug information.

  Returns:
    list: A list of dictionaries, each containing "Text" and "Coords" keys.
  '''

  if (not os.path.exists(xmlFile)):
    raise FileNotFoundError(f"XML file not found: {xmlFile}")

  # Parse the XML file into an ElementTree object.
  tree = ET.parse(xmlFile)
  # Get the root element of the parsed XML tree.
  root = tree.getroot()
  # Initialize the list that will hold parsed annotations.
  anList = []  # List of annotations.

  # Find all Annotation elements anywhere in the XML tree.
  annotations = root.findall(".//Annotation")
  if (verbose):
    # Print the number of top-level Annotation elements found.
    print("Number of annotations: ", len(annotations))

  # If no annotations were found, return the empty list.
  if (len(annotations) == 0):
    return anList

  # Iterate over each Annotation element found in the XML.
  for annotation in annotations:
    # Find all Region elements inside the current Annotation element.
    regions = annotation.findall(".//Region")
    if (verbose):
      # Print how many Region elements were found in this Annotation.
      print("- Number of regions: ", len(regions))
    # Iterate over each Region element inside the current Annotation.
    for region in regions:
      if (verbose):
        # Print the textual label associated with the current Region.
        print("-- Region Text: ", region.attrib["Text"])
      # Find all Vertex elements that define the polygon of the current Region.
      vertices = region.findall(".//Vertex")
      if (verbose):
        # Print how many Vertex elements define this Region.
        print("-- Number of vertices: ", len(vertices))

      # Build a list of integer (x, y) coordinate tuples from the Vertex attributes.
      coords = [
        (
          int(float(vertex.attrib["X"])),
          int(float(vertex.attrib["Y"]))
        )
        for vertex in vertices
      ]

      # Append a dictionary with the Region text and coordinates to the annotations list.
      anList.append(
        {
          "Text"  : region.attrib["Text"],
          "Coords": coords
        }
      )

  return anList


def ExtractWSIRegion(slide, region):
  r'''
  Extract a region of interest (ROI) from a whole-slide image (WSI) using OpenSlide.
  The extracted region is from the highest resolution level (level 0) and is masked
  according to the polygon defined by the annotation.

  Parameters:
    slide (openslide.OpenSlide or slide-like): The OpenSlide object representing the WSI or an object with a `read_region` method and `dimensions` attribute.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples.

  Returns:
    tuple: A tuple containing:
      - regionImage (numpy.ndarray): The extracted region image as a NumPy RGB array (H,W,3) uint8.
      - regionMask (numpy.ndarray): The binary mask for the region as a NumPy uint8 array (H,W) with 0/255 values.
      - roi (numpy.ndarray): The extracted region of interest (ROI) as a NumPy RGB array (H,W,3) uint8.
  '''

  # Validate the input region dictionary.
  if (region is None):
    raise ValueError("Region cannot be None.")
  if ("Coords" not in region):
    raise KeyError("Region dictionary must contain `Coords` key.")

  # Validate that the slide is an OpenSlide object.
  if (not isinstance(slide, openslide.OpenSlide)):
    raise TypeError("Slide must be an OpenSlide object.")
  if (getattr(slide, "closed", False)):
    raise ValueError("Slide is closed.")

  # Extract the list of coordinate tuples for the selected region.
  regionCoords = region["Coords"]
  # Ensure there are enough points to form a polygon.
  if (not regionCoords) or (len(regionCoords) < 1):
    raise ValueError("Region `Coords` must contain at least one (x,y) point.")

  # Build lists of x and y coordinates from the region coordinates.
  regionX = [int(x) for x, y in regionCoords]
  regionY = [int(y) for x, y in regionCoords]

  # Compute inclusive bounding box for the region in pixels (add +1 to include boundary pixels).
  minX = min(regionX)
  maxX = max(regionX)
  minY = min(regionY)
  maxY = max(regionY)
  regionWidth = maxX - minX + 1
  regionHeight = maxY - minY + 1

  # Reject degenerate boxes as they cannot form valid masks or images.
  if ((regionWidth <= 0) or (regionHeight <= 0)):
    raise ValueError("Computed region width/height must be positive.")

  # Shift the region coordinates so the polygon starts at (0,0) for mask creation.
  regionXShifted = [x - minX for x in regionX]
  regionYShifted = [y - minY for y in regionY]

  # Combine shifted x and y lists into a list of (x,y) tuples for the polygon.
  regionCoordsShifted = [(x, y) for x, y in zip(regionXShifted, regionYShifted)]

  # Convert the polygon coordinate list to a NumPy array of type int32 for OpenCV.
  regionCoordsShifted = np.array(regionCoordsShifted, np.int32)

  # Create an empty mask array of zeros with the region bounding box shape and uint8 dtype.
  regionMask = np.zeros((regionHeight, regionWidth), dtype=np.uint8)

  # Fill the polygon area on the mask with 255 to create a binary mask.
  cv2.fillPoly(regionMask, [regionCoordsShifted], 255)

  # Read the region image from the slide at level 0 using the bounding box top-left corner and size.
  regionImage = slide.read_region(
    (minX, minY),  # Top left corner.
    0,
    (regionWidth, regionHeight),  # Width x Height.
  )

  # Convert the PIL.Image returned by read_region to a NumPy uint8 array.
  regionImage = np.array(regionImage).astype(np.uint8)
  # Convert the image from RGBA to RGB color space for display.
  regionImage = cv2.cvtColor(regionImage, cv2.COLOR_RGBA2RGB)
  # Apply the mask to the region image to isolate the ROI using bitwise_and.
  roi = cv2.bitwise_and(regionImage, regionImage, mask=regionMask)
  # Convert the ROI from RGBA to RGB in case the image still contains alpha.
  roi = cv2.cvtColor(roi, cv2.COLOR_RGBA2RGB)

  # Return a tuple containing the region image, mask, and extracted ROI.
  return regionImage, regionMask, roi


def ExtractPyramidalWSITiles(
    slide,
    x=0,
    y=0,
    width=512,
    height=512,
):
  # Get the number of pyramid levels in the slide.
  slideLevels = slide.level_count

  # Create a new matplotlib figure to plot the regions from each level.
  plt.figure()

  # A dictionary to hold the extracted tiles for each level, keyed by level index.
  tiles = {}

  # Iterate over each level in the slide pyramid.
  for slideLevel in range(slideLevels):
    # Calculate the downsample ratio relative to the highest resolution level.
    dRatio = int(slide.level_downsamples[slideLevel] / slide.level_downsamples[0])

    # Calculate the horizontal offset factor used to center the crop at lower resolutions.
    factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
    # Calculate the vertical offset factor used to center the crop at lower resolutions.
    factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

    # Compute the new x-coordinate for the region at the current level.
    xNew = x - dRatio * factorWidth
    # Compute the new y-coordinate for the region at the current level.
    yNew = y - dRatio * factorHeight

    # print(f"Level {i}: Downsample Ratio={dRatio}, xNew={xNew}, yNew={yNew}")

    # Read a region from the slide at the given level and coordinates.
    regionSlide = slide.read_region(
      (xNew, yNew),
      slideLevel,
      (width, height),
    )
    # Convert the returned PIL image to RGB mode.
    regionSlide = regionSlide.convert("RGB")
    # Convert the PIL image to a NumPy array for manipulation and display.
    regionSlide = np.array(regionSlide)

    # Select the subplot for displaying the full region at this pyramid level.
    plt.subplot(2, slideLevels, slideLevel + 1)
    # Render the region image in the subplot.
    plt.imshow(regionSlide)
    # Disable axis ticks and labels for the image subplot.
    plt.axis("off")
    # Adjust subplot layout to minimize overlaps.
    plt.tight_layout()
    # Set a title for the subplot including level and downsample factor.
    plt.title(f"Level {slideLevel} ({dRatio}x)")

    # Crop the rendered region to verify the corresponding area at the current level.
    verify = regionSlide[
      factorHeight:factorHeight + height // dRatio,
      factorWidth:factorWidth + width // dRatio,
    ]

    # Select the subplot for displaying the verification crop.
    plt.subplot(2, slideLevels, 3 + slideLevel + 1)
    # Render the verification crop in the subplot.
    plt.imshow(verify)
    # Disable axis ticks and labels for the verification subplot.
    plt.axis("off")
    # Adjust layout for the verification subplot.
    plt.tight_layout()
    # Set a title for the verification subplot indicating the level verified.
    plt.title(f"Cropped to Verify Level {slideLevel}")

    # Store the extracted tile for the current level in the tiles dictionary.
    tiles[slideLevel] = regionSlide

  # Get the current figure to return it for display or saving.
  figToReturn = plt.gcf()

  return tiles, figToReturn


def PrepareAnnotationsForLevel(annotation, dFactor=1.0):
  r'''
  Map annotation coordinates from one pyramid level to another by applying a downsample factor.

  Parameters:
    annotation (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples.
    dFactor (float): The downsample factor to apply to the coordinates (default is 1.0 for no change).

  Returns:
    dict: A new annotation dictionary with the same "Text" and scaled "Coords".
  '''

  if (annotation is None):
    raise ValueError("Annotation cannot be None.")
  if ("Coords" not in annotation):
    raise KeyError("Annotation dictionary must contain `Coords` key.")

  # Extract the original coordinates from the annotation.
  originalCoords = annotation["Coords"]

  # Scale the coordinates by the downsample factor.
  mappedCoords = [
    (
      int(float(a) / dFactor),
      int(float(b) / dFactor)
    )
    for (a, b) in originalCoords
  ]

  # Shift the mapped coordinates so the polygon starts at (0,0) for mask creation.
  minX = min(x for x, y in mappedCoords)
  minY = min(y for x, y in mappedCoords)
  maxX = max(x for x, y in mappedCoords)
  maxY = max(y for x, y in mappedCoords)
  shiftedCoords = [
    (x - minX, y - minY)
    for (x, y) in mappedCoords
  ]

  mask = np.zeros((maxY - minY, maxX - minX))
  mask = cv2.fillPoly(mask, [np.array(shiftedCoords, np.int32)], 255)

  # Pad the mask to match the size of the base mask.
  baseWidth = int((maxX - minX) * dFactor)
  baseHeight = int((maxY - minY) * dFactor)
  padX = (baseWidth - mask.shape[1]) // 2
  padY = (baseHeight - mask.shape[0]) // 2
  mask = cv2.copyMakeBorder(
    mask,
    top=padY,
    bottom=padY,
    left=padX,
    right=padX,
    borderType=cv2.BORDER_CONSTANT,
    value=0
  )

  # Update the shifted coordinates to account for the padding added to the mask.
  shiftedCoords = [
    (x + padX, y + padY)
    for (x, y) in shiftedCoords
  ]

  # Return a new annotation dictionary with the same text and mapped coordinates.
  return {
    "Text"         : annotation.get("Text", ""),
    "Coords"       : mappedCoords,
    "MinX"         : minX,
    "MinY"         : minY,
    "MaxX"         : maxX,
    "MaxY"         : maxY,
    "Width"        : maxX - minX,
    "Height"       : maxY - minY,
    "ShiftedCoords": shiftedCoords,
    "dFactor"      : dFactor,
    "Mask"         : mask.astype(np.uint8),
  }


def ExtractRegionTiles(
    slide,
    region,
    width=512,
    height=512,
    overlapWidth=0,
    overlapHeight=0,
    storageDir=None,
    maxTiles=None,
    addPlots=True,
    prefix="",
    blackRatioThreshold=0.90,
    removeBackgroundTiles=True,
    convertBlackToWhite=True,
):
  r'''
  Extract tiles from a specified region of a whole-slide image (WSI) across all pyramid levels,
  applying annotation masks and saving results. The function handles the mapping of annotations
  to each level, extracts tiles, applies masks, and optionally saves the tiles, masks, and ROIs to disk.

  Parameters:
    slide (openslide.OpenSlide): The OpenSlide object representing the WSI.
    region (dict): A dictionary with a "Coords" key containing a list of (x, y) tuples representing the annotation polygon.
    width (int): The width of the tiles to extract in pixels (default is 512).
    height (int): The height of the tiles to extract in pixels (default is 512).
    overlapWidth (int): The horizontal overlap between tiles in pixels (default is 0).
    overlapHeight (int): The vertical overlap between tiles in pixels (default is 0).
    storageDir (str or None): The directory path to save the extracted tiles, masks, and ROIs. If None, no files will be saved (default is None).
    maxTiles (int or None): The maximum number of tiles to extract for the region. If None, all tiles will be extracted (default is None).
    addPlots (bool): Whether to create and save plots visualizing the tiles, masks, and ROIs (default is True).
    prefix (str): A string prefix to add to saved file names for organization (default is an empty string).
    blackRatioThreshold (float): The maximum allowed ratio of black pixels in a tile to be considered valid (default is 0.90). Tiles with a higher ratio will be skipped.
    removeBackgroundTiles (bool): Whether to skip tiles that are considered background based on the black pixel ratio (default is True).
    convertBlackToWhite (bool): Whether to convert black pixels to white in the ROI before background analysis to avoid skewing metrics (default is True).
  '''

  # Create output directories when a storage directory is provided.
  if (storageDir is not None):
    if (addPlots):
      # Compose the plots directory path and ensure it exists.
      plotsDir = os.path.join(storageDir, "Plots")
      os.makedirs(plotsDir, exist_ok=True)
    else:
      plotsDir = None
    # Compose the tiles directory path and ensure it exists.
    tilesDir = os.path.join(storageDir, "Tiles")
    os.makedirs(tilesDir, exist_ok=True)
    # Compose the masks directory path and ensure it exists.
    masksDir = os.path.join(storageDir, "Masks")
    os.makedirs(masksDir, exist_ok=True)
    # Compose the ROIs directory path and ensure it exists.
    roisDir = os.path.join(storageDir, "ROIs")
    os.makedirs(roisDir, exist_ok=True)
    # Pre-create subdirectories for each pyramid level for tiles, masks, and ROIs.
    for level in range(slide.level_count):
      os.makedirs(os.path.join(tilesDir, f"Level_{level}"), exist_ok=True)
      os.makedirs(os.path.join(masksDir, f"Level_{level}"), exist_ok=True)
      os.makedirs(os.path.join(roisDir, f"Level_{level}"), exist_ok=True)
  else:
    plotsDir = tilesDir = masksDir = roisDir = None

  # Initialize a dictionary to hold mapping data for all levels.
  mappingData = {}
  # Build mapping data for each pyramid level by preparing annotations for that level.
  for level in range(slide.level_count):
    # Compute the integer downsample factor relative to level 0.
    dFactor = int(slide.level_downsamples[level] / slide.level_downsamples[0])
    # Prepare the annotation scaled/mapped for the current level.
    annotation = PrepareAnnotationsForLevel(region, dFactor)
    # Store the prepared annotation into the mapping dictionary keyed by the level.
    mappingData[level] = annotation

  # Extract the start coordinates and dimensions of the region at the base level.
  regionStartX = mappingData[0]["MinX"]
  regionStartY = mappingData[0]["MinY"]
  regionWidth = mappingData[0]["Width"]
  regionHeight = mappingData[0]["Height"]
  category = mappingData[0]["Text"]

  xProgressBar = tqdm.tqdm(
    range(regionStartX, regionStartX + regionWidth, width - overlapWidth),
    desc="Processing X-axis",
    position=0,
  )
  yProgressBar = tqdm.tqdm(
    range(regionStartY, regionStartY + regionHeight, height - overlapHeight),
    desc="Processing Y-axis",
    leave=False,
    position=1,
  )
  # Initialize a counter to keep track of the number of tiles processed (optional, can be used for maxTiles limit).
  counter = 0
  for x in xProgressBar:
    for y in yProgressBar:
      startX = x - regionStartX
      startY = y - regionStartY

      # Extract pyramidal tiles for the current window and receive the plotting figure.
      tiles, fig1 = ExtractPyramidalWSITiles(
        slide,
        x=x,
        y=y,
        width=width,
        height=height,
      )
      # Close the temporary figure to free-associated resources.
      plt.close(fig1)
      plt.gcf().clear()  # Clear the current figure to reset the plotting state for the next iteration.

      # Create a shapely polygon for the base-level annotation to test intersection with the tile.
      baseCoordsPolygon = Polygon(mappingData[0]["ShiftedCoords"])
      # Define the polygon for the current tile region in the region-local coordinate space.
      tileRegion = Polygon([
        (startX, startY),
        (startX + width, startY),
        (startX + width, startY + height),
        (startX, startY + height),
      ])
      # Skip this tile if it does not intersect with the annotation polygon.
      if (not baseCoordsPolygon.intersects(tileRegion)):
        # print(f"Tile at x: {x}, y: {y} does not intersect with annotation region. Skipping.")
        continue

      # Prepare a plotting figure if storage is enabled so we can visualize results.
      if (addPlots and storageDir is not None):
        plt.figure(figsize=(12, 3 * slide.level_count))

      whatToStore = {}

      # Iterate over each pyramid level to crop masks and produce ROIs for saving/plotting.
      for level in range(slide.level_count):
        # Retrieve the tile image for the current level from the extracted tiles.
        levelTile = tiles[level]
        # Retrieve the precomputed mask for the current level from mappingData.
        levelMask = mappingData[level]["Mask"]
        # Retrieve the shifted coordinates used to align the mask for cropping.
        levelShiftedCoords = mappingData[level]["ShiftedCoords"]
        # Compute the minimum x coordinate of the shifted coordinates for the mask alignment.
        levelStartX = min(coord[0] for coord in levelShiftedCoords)
        # Compute the minimum y coordinate of the shifted coordinates for the mask alignment.
        levelStartY = min(coord[1] for coord in levelShiftedCoords)

        # Compute the downsample ratio integer for the current level relative to level 0.
        dRatio = int(slide.level_downsamples[level] / slide.level_downsamples[0])

        # Calculate the width padding factor to center crops at lower resolutions.
        factorWidth = int((width / 2.0) * (1.0 - (1.0 / dRatio)))
        # Calculate the height padding factor to center crops at lower resolutions.
        factorHeight = int((height / 2.0) * (1.0 - (1.0 / dRatio)))

        # Compute the x coordinate in the level mask coordinate space for cropping.
        levelX = levelStartX - factorWidth + (startX // dRatio)
        # Compute the y coordinate in the level mask coordinate space for cropping.
        levelY = levelStartY - factorHeight + (startY // dRatio)
        # Crop the mask tile from the full level mask using the computed coordinates and the requested size.
        levelMaskTile = levelMask[levelY:levelY + height, levelX:levelX + width]
        # Compute padding values needed to center the mask tile inside the level tile if sizes differ.
        padX = (levelTile.shape[1] - levelMaskTile.shape[1]) // 2
        padY = (levelTile.shape[0] - levelMaskTile.shape[0]) // 2
        # Pad the mask tile so it matches the tile image size using a constant zero border.
        levelMaskTile = cv2.copyMakeBorder(
          levelMaskTile,  # Input mask tile to be padded.
          top=padY,  # Number of pixels to pad on the top of the mask tile.
          bottom=padY,  # Number of pixels to pad on the bottom of the mask tile.
          left=padX,  # Number of pixels to pad on the left of the mask tile.
          right=padX,  # Number of pixels to pad on the right of the mask tile.
          borderType=cv2.BORDER_CONSTANT,  # Type of border to use for padding (constant value).
          value=0,  # The constant value to use for padding (0 for black).
        )
        # Ensure the padded mask tile is of type uint8 for proper masking operations.
        levelMaskTile = levelMaskTile.astype(np.uint8)

        if ((levelMaskTile.shape[0] != levelTile.shape[0]) or (levelMaskTile.shape[1] != levelTile.shape[1])):
          # Close the created figure.
          plt.close()
          whatToStore = {}
          break

        blackRatio = np.sum(levelMaskTile == 0) / levelMaskTile.size
        if ((level == 0) and (blackRatio > blackRatioThreshold)):
          # Close the created figure.
          plt.close()
          whatToStore = {}
          break

        # Compute the masked ROI by applying the binary mask to the tile image using a bitwise AND.
        levelROI = cv2.bitwise_and(levelTile, levelTile, mask=levelMaskTile)

        if (convertBlackToWhite):
          # Convert black pixels to white in the ROI.
          levelROI[levelROI == 0] = 255

        if ((level == 0) and (removeBackgroundTiles)):
          isBackground, metrics = IsBackgroundTile(
            None,
            image=levelROI.copy(),
            entropyThreshold=5.5,
            colorVarianceThreshold=1500,
            tissueAreaThreshold=0.20,
            convertBlackToWhite=convertBlackToWhite,
          )
          if (isBackground):
            # Close the created figure.
            plt.close()
            whatToStore = {}
            break

        whatToStore[level] = {
          "Tile": levelTile,
          "Mask": levelMaskTile,
          "ROI" : levelROI,
        }

      # Check if we have valid data to store for all levels before attempting to save or plot.
      if (whatToStore):
        for level in range(slide.level_count):
          levelTile = whatToStore[level]["Tile"]
          levelMaskTile = whatToStore[level]["Mask"]
          levelROI = whatToStore[level]["ROI"]

          # Save tile, mask, and ROI images to disk when storage is enabled.
          if (storageDir is not None):
            imgName = f"{level}_{x}_{y}_{width}x{height}_{overlapWidth}x{overlapHeight}"
            if (prefix):
              imgName = f"{prefix}_{imgName}"
            os.makedirs(os.path.join(tilesDir, f"Level_{level}", category), exist_ok=True)
            os.makedirs(os.path.join(masksDir, f"Level_{level}", category), exist_ok=True)
            os.makedirs(os.path.join(roisDir, f"Level_{level}", category), exist_ok=True)
            cv2.imwrite(os.path.join(tilesDir, f"Level_{level}", category, f"{imgName}.jpg"), levelTile)
            cv2.imwrite(os.path.join(masksDir, f"Level_{level}", category, f"{imgName}.jpg"), levelMaskTile)
            cv2.imwrite(os.path.join(roisDir, f"Level_{level}", category, f"{imgName}.jpg"), levelROI)

          # When storage is enabled, plot the tile, mask, overlay, and ROI for visual inspection.
          if (addPlots and storageDir is not None):
            plt.subplot(slide.level_count, 4, 1 + level * 4)
            plt.imshow(levelTile)
            plt.title("Tile")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 2 + level * 4)
            plt.imshow(levelMaskTile, cmap="gray")
            plt.title("Mask Tile")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 3 + level * 4)
            plt.imshow(levelTile)
            plt.imshow(levelMaskTile, alpha=0.5, cmap="jet")
            plt.title("Tile with Annotation Overlay")
            plt.axis("off")
            plt.subplot(slide.level_count, 4, 4 + level * 4)
            plt.imshow(levelROI)
            plt.title("ROI (Masked Tile)")
            plt.axis("off")
      else:
        # print(f"Tile at x: {x}, y: {y} has invalid mask or ROI. Skipping storage and plotting.")
        continue

      # When storage is enabled, finalize and save the plotted figure for the current tile.
      if (addPlots and storageDir is not None):
        imgName = f"{x}_{y}_{width}x{height}_{overlapWidth}x{overlapHeight}"
        if (prefix):
          imgName = f"{prefix}_{imgName}"
        os.makedirs(os.path.join(plotsDir, category), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plotsDir, category, f"{imgName}.png"), dpi=300, bbox_inches="tight")
        plt.close("all")
        plt.gcf().clear()

      counter += 1
      if ((maxTiles is not None) and (counter >= maxTiles)):
        print(f"Reached maximum tile limit of {maxTiles}. Stopping extraction.")
        return


def IsBackgroundTile(
    imagePath,  # Path to the tile image to analyze for background detection.
    image=None,  # Optional pre-loaded image as a NumPy array (H,W,3) uint8. If provided, imagePath will be ignored.
    # Threshold for Shannon entropy to detect uniformity. Adjust based on the expected variability in tissue tiles.
    entropyThreshold=5.5,
    # Threshold for color variance to detect lack of color diversity. Adjust based on the expected variability in tissue tiles.
    colorVarianceThreshold=1500,
    tissueAreaThreshold=0.20,  # Minimum ratio of tissue area to total area to consider the tile as non-background.
    convertBlackToWhite=True,  # Convert black pixels to white before analysis to avoid skewing the metrics.
):
  '''
  Detect background tiles using multiple criteria suitable for non-black backgrounds.

  Parameters:
    imagePath: Path to the tile image.
    image: Optional pre-loaded image as a NumPy array (H,W,3) uint8. If provided, imagePath will be ignored.
    entropyThreshold: Threshold for Shannon entropy to detect uniformity.
    colorVarianceThreshold: Threshold for color variance to detect lack of color diversity.
    tissueAreaThreshold: Threshold for the ratio of tissue area to total area.
    convertBlackToWhite: Whether to convert black pixels to white before analysis (default is True).

  Returns:
    bool: True if the tile is considered background, False otherwise.
    dict: A dictionary containing the computed metrics for debugging and analysis.
  '''

  import cv2
  import numpy as np
  from skimage.filters import threshold_otsu
  from skimage.measure import shannon_entropy

  if (image is None):
    if (not os.path.exists(imagePath)):
      raise FileNotFoundError(f"Image file not found: {imagePath}")
    image = cv2.imread(imagePath)

  # Convert black pixels to white to handle non-black backgrounds (e.g., white background in H&E slides).
  if (image is None):
    raise ValueError(f"Failed to load image from path: {imagePath}")

  if (convertBlackToWhite):
    image[image == 0] = 255

  # Convert to different color spaces for analysis.
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

  # 1. ENTROPY ANALYSIS (detects uniformity).
  entropyValue = shannon_entropy(gray)

  # 2. COLOR VARIANCE (H&E has characteristic pink/purple colors).
  colorVariance = np.var(image)

  # 3. TISSUE DETECTION using Otsu thresholding.
  # Invert since tissue is typically darker than background.
  thresh = threshold_otsu(gray)
  binary = gray < thresh
  tissueRatio = np.sum(binary) / binary.size

  # 4. SATURATION CHECK (H&E stained tissue has color saturation).
  saturation = hsv[:, :, 1]
  meanSaturation = np.mean(saturation)

  # 5. TEXTURE ANALYSIS (Laplacian variance).
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  textureVariance = np.var(laplacian)

  # Decision logic - tile is background if MOST criteria indicate background.
  backgroundScore = 0

  if (entropyValue < entropyThreshold):
    backgroundScore += 1
  if (colorVariance < colorVarianceThreshold):
    backgroundScore += 1
  if (tissueRatio < tissueAreaThreshold):
    backgroundScore += 1
  if (meanSaturation < 20):  # Low saturation = grayscale/white background.
    backgroundScore += 1
  if (textureVariance < 100):  # Low texture = smooth background.
    backgroundScore += 1

  # Consider background if 3 or more criteria agree.
  isBackground = backgroundScore >= 3

  return isBackground, {
    'entropy'        : entropyValue,
    'colorVariance'  : colorVariance,
    'tissueRatio'    : tissueRatio,
    'meanSaturation' : meanSaturation,
    'textureVariance': textureVariance,
    'backgroundScore': backgroundScore,
  }


if (__name__ == "__main__"):
  # Define a placeholder path for a WSI slide that the user should replace with a real path.
  slidePath = "PATH/TO/SLIDE.svs"

  # Only attempt to open the slide when the provided path exists on disk.
  if (os.path.exists(slidePath)):
    # Open the slide using the provided helper function.
    slide = ReadWSIViaOpenSlide(slidePath)
    # Print a short description of the opened slide object.
    print(f"Opened slide: {slidePath} -> {slide}.")
  else:
    # Inform the user that the example was skipped due to a missing slide file.
    print(f"Skipping ReadWSIViaOpenSlide example; file not found: {slidePath}.")

  # Define a placeholder path for a GeoJSON annotation file that the user should replace.
  annotationFile = "PATH/TO/ANNOTATIONS.geojson"

  # Only attempt to read the annotations file when it exists on disk.
  if (os.path.exists(annotationFile)):
    # Read the annotations from the GeoJSON file using the helper function.
    annotationsExample = ReadGeoJSONAnnotations(annotationFile)
    # Print the number of annotations that were read for verification.
    print(f"ReadGeoJSONAnnotations returned {len(annotationsExample)} annotations from {annotationFile}.")
  else:
    # Inform the user that the GeoJSON example was skipped due to a missing file.
    print(f"Skipping ReadGeoJSONAnnotations example; file not found: {annotationFile}.")

  # Define a placeholder path for a regular image to demonstrate polygon drawing.
  imagePath = "PATH/TO/IMAGE.jpg"

  # Only attempt to draw a polygon when the image file exists.
  if (os.path.exists(imagePath)):
    # Read the image from disk using OpenCV.
    imgArr = cv2.imread(imagePath)
    # Define a sample polygon using CamelCase dict keys and integers for coordinates.
    polygonCoords = [(10, 10), (200, 10), (200, 200), (10, 200)]
    # Draw the polygon on the loaded image using the helper function.
    imgWithPoly = DrawPolygonOnImage(imgArr, polygonCoords, outlineColor=(0, 255, 0), fillColor=None, width=3)
    # Save the resulting image to a placeholder output path for inspection.
    outputPolyPath = "PATH/TO/OUTPUT_Polygon.jpg"
    # Persist the generated image with the polygon overlay using OpenCV.
    cv2.imwrite(outputPolyPath, imgWithPoly)
    # Print a message indicating where the polygon example was stored.
    print(f"Saved polygon example to: {outputPolyPath}.")
  else:
    # Inform the user that the polygon drawing example was skipped due to a missing image.
    print(f"Skipping DrawPolygonOnImage example; file not found: {imagePath}.")

  # Define placeholder paths for WSI patch extraction that the user should adapt.
  wsiFilePath = "PATH/TO/SLIDE_FOR_PATCHES.svs"
  geojsonForPatches = "PATH/TO/PATCHES_ANNOTATIONS.geojson"
  patchesOutputDir = "PATH/TO/PATCHES_OUTPUT_DIR"

  # Only attempt patch extraction when both slide and annotations exist.
  if (os.path.exists(wsiFilePath) and os.path.exists(geojsonForPatches)):
    # Open the WSI slide via OpenSlide for patch extraction.
    wsiSlide = ReadWSIViaOpenSlide(wsiFilePath)
    # Read annotations for patch extraction from the GeoJSON file.
    patchAnnotations = ReadGeoJSONAnnotations(geojsonForPatches)
    # Call the patch extraction helper with conservative defaults.
    ExtractPatchesFromWSI(
      wsiSlide, wsiFilePath, patchAnnotations, patchesOutputDir,
      patchSize=(256, 256),
      overlap=(0, 0),
      maxNumPatchesPerAnnotation=10,
      label="ExampleLabel"
    )
    # Print a message indicating where patches were stored.
    print(f"ExtractPatchesFromWSI saved patches to: {patchesOutputDir}.")
  else:
    # Inform the user that the patch extraction example was skipped due to missing files.
    print(f"Skipping ExtractPatchesFromWSI example; required files not found: {wsiFilePath}, {geojsonForPatches}.")

  # Define placeholder paths for a pair of slides to demonstrate alignment/tiling.
  heSlidePath = "PATH/TO/HE_SLIDE.svs"
  mtSlidePath = "PATH/TO/MT_SLIDE.svs"
  alignmentStorage = "PATH/TO/ALIGNMENT_STORAGE"

  # Only attempt tile alignment when both HE and MT slides exist.
  if (os.path.exists(heSlidePath) and os.path.exists(mtSlidePath)):
    # Run the tile extraction and alignment handler with visualization disabled for speed.
    TileExtractionAlignmentHandler(
      heSlidePath, mtSlidePath, alignmentStorage,
      patchesPerSlide=100,
      targetSize=(256, 256),
      regionSize=(1024, 1024),
      overlapSize=(128, 128),
      doPlotting=False,
      verbose=True
    )
    # Print a message indicating the alignment handler was invoked.
    print(f"Invoked TileExtractionAlignmentHandler for HE:{heSlidePath} and MT:{mtSlidePath}.")
  else:
    # Inform the user that the alignment example was skipped due to missing slides.
    print(f"Skipping TileExtractionAlignmentHandler example; required slides not found: {heSlidePath}, {mtSlidePath}.")

  # Define placeholder folders for Free Form Deformation demonstration.
  heFolder = "PATH/TO/HE_IMAGES_FOLDER"
  mtFolder = "PATH/TO/MT_IMAGES_FOLDER"
  heDeformedFolder = "PATH/TO/HE_DEFORMED_OUTPUT"

  # Only attempt FFD when both source folders exist.
  if (os.path.exists(heFolder) and os.path.exists(mtFolder)):
    # Run the Free Form Deformation handler with default parameters and no plotting.
    FreeFormDeformationHandler(heFolder, mtFolder, heDeformedFolder, doPlotting=False, verbose=False)
    # Print a message indicating the FFD process was invoked.
    print(f"Invoked FreeFormDeformationHandler for HE folder: {heFolder}.")
  else:
    # Inform the user that the FFD example was skipped due to missing folders.
    print(f"Skipping FreeFormDeformationHandler example; required folders not found: {heFolder}, {mtFolder}.")

  # Define placeholder paths for random tile extraction from images demonstration.
  labelsCsv = "PATH/TO/LABELS.csv"
  slidesDirectory = "PATH/TO/SLIDES_DIR"
  tilesOutputDirectory = "PATH/TO/TILES_OUTPUT_DIR"

  # Only attempt random tile extraction when the labels CSV exists.
  if (os.path.exists(labelsCsv)):
    # Call the extract random tiles helper with small sample size for quick testing.
    results = ExtractRandomTilesFromImages(
      labelsCsv,
      slidesDir=slidesDirectory,
      outputDir=tilesOutputDirectory,
      targetShape=(256, 256),
      numOfTiles=10,
      verbose=True
    )
    # Print the summary of extracted tiles returned by the helper.
    print(f"ExtractRandomTilesFromImages results: {results}.")
  else:
    # Inform the user that the random tile example was skipped due to missing labels file.
    print(f"Skipping ExtractRandomTilesFromImages example; labels file not found: {labelsCsv}.")

  # Define a placeholder path for a BACH XML annotation file to demonstrate XML parsing.
  bachXml = "PATH/TO/BACH_ANNOTATIONS.xml"

  # Only attempt to parse BACH annotations when the XML file exists.
  if (os.path.exists(bachXml)):
    # Extract regions from the BACH-style XML using the helper function.
    bachRegions = ExtractBACHAnnotationsFromXML(bachXml, verbose=False)
    # Print the number of regions parsed from the XML file for verification.
    print(f"ExtractBACHAnnotationsFromXML returned {len(bachRegions)} regions from {bachXml}.")
  else:
    # Inform the user that the BACH XML example was skipped due to a missing file.
    print(f"Skipping ExtractBACHAnnotationsFromXML example; file not found: {bachXml}.")

  # Define a placeholder path for testing `ExtractWSIRegion` using an existing slide and a simple region example.
  regionSlidePath = "PATH/TO/REGION_SLIDE.svs"
  # Construct a sample region dictionary using CamelCase keys expected by the helper functions.
  regionExample = {"Text": "ExampleRegion", "Coords": [(100, 100), (400, 100), (400, 400), (100, 400)]}

  # Only attempt to extract the WSI region when the slide file exists.
  if (os.path.exists(regionSlidePath)):
    # Open the slide used for the region extraction example.
    regionSlide = ReadWSIViaOpenSlide(regionSlidePath)
    # Extract the region image, mask, and ROI using the helper function.
    regionImage, regionMask, regionROI = ExtractWSIRegion(regionSlide, regionExample)
    # Print basic info about the returned arrays for confirmation.
    print(
      f"ExtractWSIRegion returned image shape: {getattr(regionImage, 'shape', None)}, "
      f"mask shape: {getattr(regionMask, 'shape', None)}."
    )
  else:
    # Inform the user that the WSI region example was skipped due to a missing slide file.
    print(f"Skipping ExtractWSIRegion example; file not found: {regionSlidePath}.")

  # Define a placeholder path for a slide to demonstrate pyramidal tile extraction.
  pyramidalSlidePath = "PATH/TO/PYRAMIDAL_SLIDE.svs"

  # Only attempt the pyramidal tiles example when the slide file exists.
  if (os.path.exists(pyramidalSlidePath)):
    # Open the pyramidal slide for tile extraction.
    pyramidalSlide = ReadWSIViaOpenSlide(pyramidalSlidePath)
    # Extract tiles at multiple pyramid levels for a sample anchor point.
    tilesDict, tilesFig = ExtractPyramidalWSITiles(pyramidalSlide, x=0, y=0, width=512, height=512)
    # Print the number of pyramid levels that were returned by the helper.
    print(f"ExtractPyramidalWSITiles returned {len(tilesDict)} levels.")
  else:
    # Inform the user that the pyramidal tiles example was skipped due to a missing slide.
    print(f"Skipping ExtractPyramidalWSITiles example; file not found: {pyramidalSlidePath}.")

  # Define a simple annotation to demonstrate PrepareAnnotationsForLevel processing.
  simpleAnnotation = {"Text": "Simple", "Coords": [(0, 0), (100, 0), (100, 100), (0, 100)]}
  # Prepare the annotation for a downsampled level using a sample dFactor.
  prepared = PrepareAnnotationsForLevel(simpleAnnotation, dFactor=2.0)
  # Print a summary of the prepared annotation to verify output keys and values.
  print(f"PrepareAnnotationsForLevel produced keys: {list(prepared.keys())}.")

  # Define a placeholder region and storage directory for region tiling demonstration.
  regionTileSlidePath = "PATH/TO/REGION_TILE_SLIDE.svs"
  regionTileStorage = "PATH/TO/REGION_TILE_OUTPUT"

  # Only attempt region tiling when the slide for tiling exists.
  if (os.path.exists(regionTileSlidePath)):
    # Open the slide that will be used for region tiling.
    regionTileSlide = ReadWSIViaOpenSlide(regionTileSlidePath)
    # Use the previously defined simpleAnnotation as the region to tile.
    ExtractRegionTiles(
      regionTileSlide,
      simpleAnnotation,
      width=512,
      height=512,
      overlapWidth=0,
      overlapHeight=0,
      storageDir=regionTileStorage
    )
    # Print a message indicating where the region tiles were stored.
    print(f"Invoked ExtractRegionTiles and stored outputs (when enabled) under: {regionTileStorage}.")
  else:
    # Inform the user that the region tiling example was skipped due to a missing slide.
    print(f"Skipping ExtractRegionTiles example; slide not found: {regionTileSlidePath}.")
