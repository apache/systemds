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
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gd


def download_data(data_dir):
  """Download raw and preprocessed data files.
  The data is downloaded from Google Drive and stored in the 'data/' directory.
  """
  print(f"Downloading the raw and preprocessed data into {data_dir}.")

  if not os.path.exists(data_dir):
    print('Downloading data directory.')
    dir_name = data_dir
    gd.download_file_from_google_drive(
      file_id='1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU',
      dest_path=dir_name,
      unzip=True,
      showsize=True
    )
  print('Data was downloaded.')


def parquet_to_csv(parquetfilename):
  print(f'got file: {parquetfilename}')
  df = pd.read_parquet(parquetfilename)
  dest_name = str(parquetfilename).replace("parquet", "csv")
  df.to_csv(dest_name)


def main():
  print("Downloading data sets for the sherlock classification network")
  data_dir = '../data/'
  download_data(data_dir + 'data.zip')
  parquet_to_csv(data_dir + "data/processed/X_train.parquet")
  parquet_to_csv(data_dir + "data/processed/X_val.parquet")
  parquet_to_csv(data_dir + "data/processed/X_test.parquet")
  parquet_to_csv(data_dir + "data/processed/y_train.parquet")
  parquet_to_csv(data_dir + "data/processed/y_val.parquet")
  parquet_to_csv(data_dir + "data/processed/y_test.parquet")

  print("This data can be used to train and test the sherlock network implemented in sherlock.dml")
  print("The X_*.csv files have to be processed with the function: sherlock::transform_values()")
  print("The y_*.csv files have to be processed with the function: \n"
        "   sherlock::transform_encode_labels() to encode the output categories to numbers.\n"
        "   sherlock::transform_apply_labels() is used to apply the above created encoding to the remaining input files.")
  print("The processed files can then be used to train the network: \n"
        "   sherlock(X_train, y_train)")


if __name__ == '__main__':
  main()
