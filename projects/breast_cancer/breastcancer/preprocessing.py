#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

"""
Preprocessing -- Predicting Breast Cancer Proliferation Scores with
Apache SystemML

This module contains functions for the preprocessing phase of the
breast cancer project.
"""

import math
import os

import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd
from pyspark.ml.linalg import Vectors
import pyspark.sql.functions as F
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk


# Open Whole-Slide Image

def open_slide(slide_num, folder, training):
  """
  Open a whole-slide image, given an image number.

  Args:
    slide_num: Slide image number as an integer.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  if training:
    filename = os.path.join(folder, "training_image_data",
                            "TUPAC-TR-{}.svs".format(str(slide_num).zfill(3)))
  else:
    # Testing images
    filename = os.path.join(folder, "testing_image_data",
                            "TUPAC-TE-{}.svs".format(str(slide_num).zfill(3)))
  slide = openslide.open_slide(filename)
  return slide


# Create Tile Generator

def create_tile_generator(slide, tile_size, overlap):
  """
  Create a tile generator for the given slide.

  This generator is able to extract tiles from the overall
  whole-slide image.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.

  Returns:
    A DeepZoomGenerator object representing the tile generator. Each
    extracted tile is a PIL Image with shape
    (tile_size, tile_size, channels).
    Note: This generator is not a true "Python generator function", but
    rather is an object that is capable of extracting individual tiles.
  """
  generator = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=True)
  return generator


# Determine 20x Magnification Zoom Level

def get_20x_zoom_level(slide, generator):
  """
  Return the zoom level that corresponds to a 20x magnification.

  The generator can extract tiles from multiple zoom levels,
  downsampling by a factor of 2 per level from highest to lowest
  resolution.

  Args:
    slide: An OpenSlide object representing a whole-slide image.
    generator: A DeepZoomGenerator object representing a tile generator.
      Note: This generator is not a true "Python generator function",
      but rather is an object that is capable of extracting individual
      tiles.

  Returns:
    Zoom level corresponding to a 20x magnification, or as close as
    possible.
  """
  highest_zoom_level = generator.level_count - 1  # 0-based indexing
  try:
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    # `mag / 20` gives the downsampling factor between the slide's
    # magnification and the desired 20x magnification.
    # `(mag / 20) / 2` gives the zoom level offset from the highest
    # resolution level, based on a 2x downsampling factor in the
    # generator.
    offset = math.floor((mag / 20) / 2)
    level = highest_zoom_level - offset
  except ValueError:
    # In case the slide magnification level is unknown, just
    # use the highest resolution.
    level = highest_zoom_level
  return level


# Generate Tile Indices For Whole-Slide Image.

def process_slide(slide_num, folder, training, tile_size, overlap):
  """
  Generate all possible tile indices for a whole-slide image.

  Given a slide number, tile size, and overlap, generate
  all possible (slide_num, tile_size, overlap, zoom_level, col, row)
  indices.

  Args:
    slide_num: Slide image number as an integer.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.

  Returns:
    A list of (slide_num, tile_size, overlap, zoom_level, col, row)
    integer index tuples representing possible tiles to extract.
  """
  # Open slide.
  slide = open_slide(slide_num, folder, training)
  # Create tile generator.
  generator = create_tile_generator(slide, tile_size, overlap)
  # Get 20x zoom level.
  zoom_level = get_20x_zoom_level(slide, generator)
  # Generate all possible (zoom_level, col, row) tile index tuples.
  cols, rows = generator.level_tiles[zoom_level]
  tile_indices = [(slide_num, tile_size, overlap, zoom_level, col, row)
                  for col in range(cols) for row in range(rows)]
  return tile_indices


# Generate Tile From Tile Index

def process_tile_index(tile_index, folder, training):
  """
  Generate a tile from a tile index.

  Given a (slide_num, tile_size, overlap, zoom_level, col, row) tile
  index, generate a (slide_num, tile) tuple.

  Args:
    tile_index: A (slide_num, tile_size, overlap, zoom_level, col, row)
      integer index tuple representing a tile to extract.
    folder: Directory in which the slides folder is stored, as a string.
      This should contain either a `training_image_data` folder with
      images in the format `TUPAC-TR-###.svs`, or a `testing_image_data`
      folder with images in the format `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.

  Returns:
    A (slide_num, tile) tuple, where slide_num is an integer, and tile
    is a 3D NumPy array of shape (tile_size, tile_size, channels) in
    RGB format.
  """
  slide_num, tile_size, overlap, zoom_level, col, row = tile_index
  # Open slide.
  slide = open_slide(slide_num, folder, training)
  # Create tile generator.
  generator = create_tile_generator(slide, tile_size, overlap)
  # Generate tile.
  tile = np.asarray(generator.get_tile(zoom_level, (col, row)))
  return (slide_num, tile)


# Filter Tile For Dimensions & Tissue Threshold

def optical_density(tile):
  """
  Convert a tile to optical density values.

  Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).

  Returns:
    A 3D NumPy array of shape (tile_size, tile_size, channels)
    representing optical density values.
  """
  tile = tile.astype(np.float64)
  #od = -np.log10(tile/255 + 1e-8)
  od = -np.log((tile+1)/240)
  return od


def keep_tile(tile_tuple, tile_size, tissue_threshold):
  """
  Determine if a tile should be kept.

  This filters out tiles based on size and a tissue percentage
  threshold, using a custom algorithm. If a tile has height &
  width equal to (tile_size, tile_size), and contains greater
  than or equal to the given percentage, then it will be kept;
  otherwise it will be filtered out.

  Args:
    tile_tuple: A (slide_num, tile) tuple, where slide_num is an
      integer, and tile is a 3D NumPy array of shape
      (tile_size, tile_size, channels) in RGB format.
    tile_size: The width and height of a square tile to be generated.
    tissue_threshold: Tissue percentage threshold.

  Returns:
    A Boolean indicating whether or not a tile should be kept for
    future usage.
  """
  slide_num, tile = tile_tuple
  if tile.shape[0:2] == (tile_size, tile_size):
    tile_orig = tile

    # Check 1
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
    tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
    tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
    tile = canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
    tile = binary_closing(tile, disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
    tile = binary_dilation(tile, disk(10))
    # Fill remaining holes within regions of tissue.
    tile = binary_fill_holes(tile)
    # Calculate percentage of tissue coverage.
    percentage = tile.mean()
    check1 = percentage >= tissue_threshold

    # Check 2
    # Convert to optical density values
    tile = optical_density(tile_orig)
    # Threshold at beta
    beta = 0.15
    tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
    tile = binary_closing(tile, disk(2))
    tile = binary_dilation(tile, disk(2))
    tile = binary_fill_holes(tile)
    percentage = tile.mean()
    check2 = percentage >= tissue_threshold

    return check1 and check2
  else:
    return False


# Generate Samples From Tile

def process_tile(tile_tuple, sample_size, grayscale):
  """
  Process a tile into a group of smaller samples.

  Cut up a tile into smaller blocks of sample_size x sample_size pixels,
  change the shape of each sample from (H, W, channels) to
  (channels, H, W), then flatten each into a vector of length
  channels*H*W.

  Args:
    tile_tuple: A (slide_num, tile) tuple, where slide_num is an
      integer, and tile is a 3D NumPy array of shape
      (tile_size, tile_size, channels).
    sample_size: The new width and height of the square samples to be
      generated.
    grayscale: Whether or not to generate grayscale samples, rather
      than RGB.

  Returns:
    A list of (slide_num, sample) tuples representing cut up tiles,
    where each sample is a 3D NumPy array of shape
    (sample_size_x, sample_size_y, channels).
  """
  slide_num, tile = tile_tuple
  if grayscale:
    tile = rgb2gray(tile)[:, :, np.newaxis]  # Grayscale
    # Save disk space and future IO time by converting from [0,1] to [0,255],
    # at the expense of some minor loss of information.
    tile = np.round(tile * 255).astype("uint8")
  x, y, ch = tile.shape
  # 1. Reshape into a 5D array of (num_x, sample_size_x, num_y, sample_size_y, ch), where
  # num_x and num_y are the number of chopped tiles on the x and y axes, respectively.
  # 2. Swap sample_size_x and num_y axes to create
  # (num_x, num_y, sample_size_x, sample_size_y, ch).
  # 3. Combine num_x and num_y into single axis, returning
  # (num_samples, sample_size_x, sample_size_y, ch).
  samples = (tile.reshape((x // sample_size, sample_size, y // sample_size, sample_size, ch))
                 .swapaxes(1,2)
                 .reshape((-1, sample_size, sample_size, ch)))
  samples = [(slide_num, sample) for sample in list(samples)]
  return samples


# Normalize staining

def normalize_staining(sample_tuple, beta=0.15, alpha=1, light_intensity=255):
  """
  Normalize the staining of H&E histology slides.

  This function normalizes the staining of H&E histology slides.

  References:
    - Macenko, Marc, et al. "A method for normalizing histology slides
    for quantitative analysis." Biomedical Imaging: From Nano to Macro,
    2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
      - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    - https://github.com/mitkovetta/staining-normalization

  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).

  Returns:
    A (slide_num, sample) tuple, where the sample is a 3D NumPy array
    of shape (H,W,C) that has been stain normalized.
  """
  # Setup.
  slide_num, sample = sample_tuple
  x = np.asarray(sample)
  h, w, c = x.shape
  x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

  # Reference stain vectors and stain saturations.  We will normalize all slides
  # to these references.  To create these, grab the stain vectors and stain
  # saturations from a desirable slide.

  # Values in reference implementation for use with eigendecomposition approach, natural log,
  # and `light_intensity=240`.
  #stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
  #max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

  # SVD w/ log10, and `light_intensity=255`.
  stain_ref = (np.array([0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629])
                 .reshape(3,2))
  max_sat_ref = np.array([0.82791151, 0.61137274]).reshape(2,1)

  # Convert RGB to OD.
  # Note: The original paper used log10, and the reference implementation used the natural log.
  #OD = -np.log((x+1)/light_intensity)  # shape (H*W, C)
  OD = -np.log10(x/light_intensity + 1e-8)

  # Remove data with OD intensity less than beta.
  # I.e. remove transparent pixels.
  # Note: This needs to be checked per channel, rather than
  # taking an average over all channels for a given pixel.
  OD_thresh = OD[np.all(OD >= beta, 1), :]  # shape (K, C)

  # Calculate eigenvectors.
  # Note: We can either use eigenvector decomposition, or SVD.
  #eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
  U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

  # Extract two largest eigenvectors.
  # Note: We swap the sign of the eigvecs here to be consistent
  # with other implementations.  Both +/- eigvecs are valid, with
  # the same eigenvalue, so this is okay.
  #top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
  top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

  # Project thresholded optical density values onto plane spanned by
  # 2 largest eigenvectors.
  proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

  # Calculate angle of each point wrt the first plane direction.
  # Note: the parameters are `np.arctan2(y, x)`
  angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

  # Find robust extremes (a and 100-a percentiles) of the angle.
  min_angle = np.percentile(angles, alpha)
  max_angle = np.percentile(angles, 100-alpha)

  # Convert min/max vectors (extremes) back to optimal stains in OD space.
  # This computes a set of axes for each angle onto which we can project
  # the top eigenvectors.  This assumes that the projected values have
  # been normalized to unit length.
  extreme_angles = np.array(
    [[np.cos(min_angle), np.cos(max_angle)],
     [np.sin(min_angle), np.sin(max_angle)]]
  )  # shape (2,2)
  stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

  # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
  if stains[0, 0] < stains[0, 1]:
    stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

  # Calculate saturations of each stain.
  # Note: Here, we solve
  #    OD = VS
  #     S = V^{-1}OD
  # where `OD` is the matrix of optical density values of our image,
  # `V` is the matrix of stain vectors, and `S` is the matrix of stain
  # saturations.  Since this is an overdetermined system, we use the
  # least squares solver, rather than a direct solve.
  sats, _, _, _ = np.linalg.lstsq(stains, OD.T)

  # Normalize stain saturations to have same pseudo-maximum based on
  # a reference max saturation.
  max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
  sats = sats / max_sat * max_sat_ref

  # Compute optimal OD values.
  OD_norm = np.dot(stain_ref, sats)

  # Recreate image.
  # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
  # not return the correct values due to the initital values being outside of [0,255].
  # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
  # same behavior as Matlab.
  #x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
  x_norm = 10**(-OD_norm) * light_intensity - 1e-8  # log10 approach
  x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
  x_norm = x_norm.astype(np.uint8)
  x_norm = x_norm.T.reshape(h,w,c)
  return (slide_num, x_norm)


def flatten_sample(sample_tuple):
  """
  Flatten a (H,W,C) sample into a (C*H*W) row vector.

  Transpose each sample from (H, W, channels) to (channels, H, W), then
  flatten each into a vector of length channels*H*W.

  Args:
    sample_tuple: A (slide_num, sample) tuple, where slide_num is an
      integer, and sample is a 3D NumPy array of shape (H,W,C).

  Returns:
    A (slide_num, sample) tuple, where the sample has been transposed
    from (H,W,C) to (C,H,W), and flattened to a vector of length
    (C*H*W).
  """
  slide_num, sample = sample_tuple
  # 1. Swap axes from (sample_size_x, sample_size_y, ch) to
  # (ch, sample_size_x, sample_size_y).
  # 2. Flatten sample into (ch*sample_size_x*sample_size_y).
  flattened_sample = sample.transpose(2,0,1).reshape(-1)
  return (slide_num, flattened_sample)


# Get Ground Truth Labels

def get_labels_df(folder):
  """
  Create a DataFrame with the ground truth labels for each slide.

  Args:
    folder: Directory containing a `training_ground_truth.csv` file
      containing the ground truth "tumor_score" and "molecular_score"
      labels for each slide.

  Returns:
    A Pandas DataFrame containing the ground truth labels for each
    slide.
  """
  filepath = os.path.join(folder, "training_ground_truth.csv")
  labels_df = pd.read_csv(filepath, names=["tumor_score", "molecular_score"], header=None)
  labels_df["slide_num"] = labels_df.index + 1  # slide numbering starts at 1
  labels_df.set_index("slide_num", drop=False, inplace=True)  # use the slide num as index
  return labels_df


# Process All Slides Into A Spark DataFrame

def preprocess(spark, slide_nums, folder="data", training=True, tile_size=1024, overlap=0,
               tissue_threshold=0.9, sample_size=256, grayscale=False, normalize_stains=True,
               num_partitions=20000):
  """
  Preprocess a set of whole-slide images.

  Preprocess a set of whole-slide images as follows:
    1. Tile the slides into tiles of size (tile_size, tile_size, 3).
    2. Filter the tiles to remove unnecessary tissue.
    3. Cut the remaining tiles into samples of size
       (sample_size, sample_size, ch), where `ch` is 1 if `grayscale`
       is true, or 3 otherwise.

  Args:
    spark: SparkSession.
    slide_nums: List of whole-slide numbers to process.
    folder: Local directory in which the slides folder and ground truth
      file is stored, as a string. This should contain a
      `training_image_data` folder with images in the format
      `TUPAC-TR-###.svs`, as well as a `training_ground_truth.csv` file
      containing the ground truth "tumor_score" and "molecular_score"
      labels for each slide.  Alternatively, the folder should contain a
      `testing_image_data` folder with images in the format
      `TUPAC-TE-###.svs`.
    training: Boolean for training or testing datasets.
    tile_size: The width and height of a square tile to be generated.
    overlap: Number of pixels by which to overlap the tiles.
    tissue_threshold: Tissue percentage threshold for filtering.
    sample_size: The new width and height of the square samples to be
      generated.
    grayscale: Whether or not to generate grayscale samples, rather
      than RGB.
    normalize_stains: Whether or not to apply stain normalization.
    num_partitions: Number of partitions to use during processing.

  Returns:
    A Spark DataFrame in which each row contains the slide number, tumor
    score, molecular score, and the sample stretched out into a Vector.
  """
  slides = spark.sparkContext.parallelize(slide_nums)
  # Create DataFrame of all tile locations and increase number of partitions
  # to avoid OOM during subsequent processing.
  tile_indices = (slides.flatMap(
      lambda slide: process_slide(slide, folder, training, tile_size, overlap)))
  # TODO: Explore computing the ideal paritition sizes based on projected number
  #   of tiles after filtering.  I.e. something like the following:
  #rows = tile_indices.count()
  #part_size = 128
  #channels = 1 if grayscale else 3
  #row_mb = tile_size * tile_size * channels * 8 / 1024 / 1024  # size of one row in MB
  #rows_per_part = round(part_size / row_mb)
  #num_parts = rows / rows_per_part
  tile_indices = tile_indices.repartition(num_partitions)
  tile_indices.cache()
  # Extract all tiles into a DataFrame, filter, cut into smaller samples, apply stain
  # normalization, and flatten.
  tiles = tile_indices.map(lambda tile_index: process_tile_index(tile_index, folder, training))
  filtered_tiles = tiles.filter(lambda tile: keep_tile(tile, tile_size, tissue_threshold))
  samples = filtered_tiles.flatMap(lambda tile: process_tile(tile, sample_size, grayscale))
  if normalize_stains:
    samples = samples.map(lambda sample: normalize_staining(sample))
  samples = samples.map(lambda sample: flatten_sample(sample))
  if training:
    # Append labels
    labels_df = get_labels_df(folder)
    samples_with_labels = (samples.map(
        lambda tup: (tup[0], int(labels_df.at[tup[0],"tumor_score"]),
                     float(labels_df.at[tup[0],"molecular_score"]), Vectors.dense(tup[1]))))
    df = samples_with_labels.toDF(["slide_num", "tumor_score", "molecular_score", "sample"])
    df = df.select(df.slide_num.astype("int"), df.tumor_score.astype("int"),
                   df.molecular_score, df["sample"])
  else:  # testing data -- no labels
    df = samples.toDF(["slide_num", "sample"])
    df = df.select(df.slide_num.astype("int"), df["sample"])
  return df


# Split Into Separate Train & Validation DataFrames Based On Slide Number

def train_val_split(spark, df, slide_nums, folder, train_frac=0.8, add_row_indices=True, seed=None,
                    debug=False):
  """
  Split a DataFrame of slide samples into training and validation sets.

  Args:
    spark: SparkSession.
    df: A Spark DataFrame in which each row contains the slide number,
    tumor score, molecular score, and the sample stretched out into
    a Vector.
    slide_nums: A list of slide numbers to sample from.
    folder: Directory containing a `training_ground_truth.csv` file
      containing the ground truth "tumor_score" and "molecular_score"
      labels for each slide.
    train_frac: Fraction of the data to assign to the training set, with
      `1-frac` assigned to the valiation set.
    add_row_indices: Boolean for whether or not to prepend an index
      column contain the row index for use downstream by SystemML.
      The column name will be "__INDEX".

  Returns:
    A Spark DataFrame in which each row contains the slide number, tumor
    score, molecular score, and the sample stretched out into a Vector.
  """
  # Create DataFrame of labels for the given slide numbers.
  labels_df = get_labels_df(folder)
  labels_df = labels_df.loc[slide_nums]

  # Randomly split slides 80%/20% into train and validation sets.
  train_nums_df = labels_df.sample(frac=train_frac, random_state=seed)
  val_nums_df = labels_df.drop(train_nums_df.index)

  train_nums = (spark.createDataFrame(train_nums_df)
                     .selectExpr("cast(slide_num as int)")
                     .coalesce(1))
  val_nums = (spark.createDataFrame(val_nums_df)
                   .selectExpr("cast(slide_num as int)")
                   .coalesce(1))

  # Note: Explicitly mark the smaller DataFrames as able to be broadcasted
  # in order to have Catalyst choose the more efficient BroadcastHashJoin,
  # rather than the costly SortMergeJoin.
  train = df.join(F.broadcast(train_nums), on="slide_num")
  val = df.join(F.broadcast(val_nums), on="slide_num")

  if debug:
    # DEBUG: Sanity checks.
    assert len(pd.merge(train_nums_df, val_nums_df, on="slide_num")) == 0
    assert train_nums.join(val_nums, on="slide_num").count() == 0
    assert train.join(val, on="slide_num").count() == 0
    #  - Check distributions.
    for pdf in train_nums_df, val_nums_df:
      print(pdf.count())
      print(pdf["tumor_score"].value_counts(sort=False))
      print(pdf["tumor_score"].value_counts(normalize=True, sort=False), "\n")
    #  - Check total number of examples in each.
    print(train.count(), val.count())
    #  - Check physical plans for broadcast join.
    print(train.explain(), val.explain())

  # Add row indices for use with SystemML.
  if add_row_indices:
    train = (train.rdd
                  .zipWithIndex()
                  .map(lambda r: (r[1] + 1, *r[0]))  # flatten & convert index to 1-based indexing
                  .toDF(['__INDEX', 'slide_num', 'tumor_score', 'molecular_score', 'sample']))
    train = train.select(train["__INDEX"].astype("int"), train.slide_num.astype("int"),
                         train.tumor_score.astype("int"), train.molecular_score, train["sample"])

    val = (val.rdd
              .zipWithIndex()
              .map(lambda r: (r[1] + 1, *r[0]))  # flatten & convert index to 1-based indexing
              .toDF(['__INDEX', 'slide_num', 'tumor_score', 'molecular_score', 'sample']))
    val = val.select(val["__INDEX"].astype("int"), val.slide_num.astype("int"),
                     val.tumor_score.astype("int"), val.molecular_score, val["sample"])

  return train, val


# Save DataFrame

def save(df, filepath, sample_size, grayscale, mode="error", format="parquet", file_size=128):
  """
  Save a preprocessed DataFrame with a constraint on the file sizes.

  Args:
    df: A Spark DataFrame.
    filepath: Hadoop-supported path at which to save `df`.
    sample_size: The width and height of the square samples.
    grayscale: Whether or not to the samples are in grayscale format,
      rather than RGB.
    mode: Specifies the behavior of `df.write.mode` when the data
      already exists.  Options include:
        * `append`: Append contents of this DataFrame to
          existing data.
        * `overwrite`: Overwrite existing data.
        * `error`: Throw an exception if data already exists.
        * `ignore`: Silently ignore this operation if data already
          exists.
    format: The format in which to save the DataFrame.
    file_size: Size in MB of each saved file.  128 MB is an
      empirically ideal size.
  """
  channels = 1 if grayscale else 3
  row_mb = sample_size * sample_size * channels * 8 / 1024 / 1024  # size of one row in MB
  rows_per_file = round(file_size / row_mb)
  df.write.option("maxRecordsPerFile", rows_per_file).mode(mode).save(filepath, format=format)

