# Import the required libraries.
import PIL, cv2, os, openslide, json
import numpy as np
from pathlib import Path
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
  if (not p.exists()):
    print(f"ERROR: Annotation file does not exist: {p}")
    return []

  try:
    with p.open("r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception as e:
    print(f"ERROR: Failed to read/parse GeoJSON '{p}': {e}")
    return []

  annotations: List[Dict[str, Any]] = []

  # Normalize to a list of features
  features = None
  if (isinstance(data, dict) and isinstance(data.get("features"), list)):
    features = data["features"]
  elif (isinstance(data, dict) and data.get("type") == "Feature"):
    features = [data]
  elif (isinstance(data, dict) and data.get("type") == "FeatureCollection"):
    features = data.get("features", [])
  else:
    # If data looks like a geometry object, wrap it as a single feature-like entry
    if (isinstance(data, dict) and ("type" in data and "coordinates" in data)):
      features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
      print(f"WARNING: GeoJSON file '{p.name}' doesn't contain features or a geometry object; returning empty list")
      return []

  for idx, feat in enumerate(features):
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
      os.path.join(thumbStoragePath, f"Slide_{os.path.basename(heSlidePath)}.png"),
      dpi=720,
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
        #     dpi=720,
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
    IsPointInsideContour(topLeftDownSampled, heContour)
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
