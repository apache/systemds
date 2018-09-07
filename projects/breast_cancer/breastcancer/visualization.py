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
Visualization -- Predicting Breast Cancer Proliferation Scores with
Apache SystemML

This module contains functions for visualizing data for the breast
cancer project.
"""
import matplotlib.pyplot as plt


def visualize_tile(tile):
  """
  Plot a tissue tile.
  
  Args:
    tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
  
  Returns:
    None
  """
  plt.imshow(tile)
  plt.show()


def visualize_sample(sample, size=256):
  """
  Plot a tissue sample.
  
  Args:
    sample: A square sample flattened to a vector of size
      (channels*size_x*size_y).
    size: The width and height of the square samples.
  
  Returns:
    None
  """
  # Change type, reshape, transpose to (size_x, size_y, channels).
  length = sample.shape[0]
  channels = int(length / (size * size))
  if channels > 1:
    sample = sample.astype('uint8').reshape((channels, size, size)).transpose(1,2,0)
    plt.imshow(sample)
  else:
    vmax = 255 if sample.max() > 1 else 1
    sample = sample.reshape((size, size))
    plt.imshow(sample, cmap="gray", vmin=0, vmax=vmax)
  plt.show()

