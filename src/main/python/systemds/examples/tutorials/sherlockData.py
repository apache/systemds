#!/usr/bin/env python3
# -------------------------------------------------------------
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
# -------------------------------------------------------------

import os
from google_drive_downloader import GoogleDriveDownloader as gd
import pandas as pd

class SherlockData:
  '''
  This data set holds data for semantic data type detection.

  The data can be used to train and test the sherlock network implemented in sherlock.dml
  The X_*.csv files have to be processed with the function: sherlock::transform_values()
  The y_*.csv files have to be processed with the function: \n
    sherlock::transform_encode_labels() to encode the output categories to numbers.\n
    sherlock::transform_apply_labels() is used to apply the above created encoding to the remaining input files.
  The processed files can then be used to train the network: \n
    sherlock(X_train, y_train)
  '''

  _base_path: str
  _processed_dir: str
  _raw_dir: str

  def __init__(self):
    self._base_path = "systemds/examples/tutorials/sherlock/"
    self._processed_dir = "data/processed/"
    self._raw_dir = "data/raw/"

  def parquet_to_csv(self, parquetfile):
    print(f'got file: {parquetfile}')
    df = pd.read_parquet(parquetfile)
    dest_name = str(parquetfile).replace("parquet", "csv")
    df.to_csv(dest_name)

  def get_train_values(self, processed):
    return self._get_values(processed=processed, type="train")

  def get_val_values(self, processed):
    return self._get_values(processed=processed, type="val")

  def get_test_values(self, processed):
    return self._get_values(processed=processed, type="test")

  def get_train_labels(self, processed):
    return self._get_labels(processed=processed, type="train")

  def get_val_labels(self, processed):
    return self._get_labels(processed=processed, type="val")

  def get_test_labels(self, processed):
    return self._get_labels(processed=processed, type="test")

  def _get_values(self, processed, type):
    filename_parquet = self._base_path
    filename_parquet += self._processed_dir + "X_{}.parquet".format(type) \
      if processed ==True else self._raw_dir + "/{}_values.parquet".format(type)

    if not os.path.exists(filename_parquet):
      self._download_data(self._base_path)
    return pd.read_parquet(filename_parquet)

  def _get_labels(self, processed, type):
    filename_parquet = self._base_path
    filename_parquet += self._processed_dir + "y_{}.parquet".format(type) \
      if processed ==True else self._raw_dir + "{}_labels.parquet".format(type)

    if not os.path.exists(filename_parquet):
      self._download_data(self._base_path)
    return pd.read_parquet(filename_parquet)

    def _download_data(self, data_dir):
      """Download raw and preprocessed data files.
      The data is downloaded from Google Drive and stored in the 'data/' directory.
      """
    print(f"Downloading the raw and preprocessed data into {data_dir}.")

    if not os.path.exists(data_dir + "data"):
      os.makedirs(data_dir, exist_ok=True)
      print('Downloading data directory.')
      filename = data_dir + "data.zip"
      gd.download_file_from_google_drive(
        file_id='1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU',
        dest_path=filename,
        unzip=True,
        showsize=True
      )

    print('Data was downloaded.')
