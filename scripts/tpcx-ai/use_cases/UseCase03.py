#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


#
import argparse
import datetime
import os
import timeit
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
from tqdm import tqdm


class UseCase03Model(object):

    def __init__(self, use_store=False, use_department=True):
        if not use_store and not use_department:
            raise ValueError(f"use_store = {use_store}, use_department = {use_department}: at least one must be True")

        self._use_store = use_store
        self._use_department = use_department
        self._models = {}
        self._min = {}
        self._max = {}

    def _get_key(self, store, department):
        if self._use_store and self._use_department:
            key = (store, department)
        elif self._use_store:
            key = store
        else:
            key = department

        return key

    def store_model(self, store: int, department: int, model, ts_min, ts_max):
        key = self._get_key(store, department)
        self._models[key] = model
        self._min[key] = ts_min
        self._max[key] = ts_max

    def get_model(self, store: int, department: int):
        key = self._get_key(store, department)
        model = self._models[key]
        ts_min = self._min[key]
        ts_max = self._max[key]
        return model, ts_min, ts_max


def load(order_path: str, lineitem_path: str, product_path: str) -> pd.DataFrame:
    order_data = pd.read_csv(order_path, parse_dates=['date'])
    lineitem_data = pd.read_csv(lineitem_path)
    product_data = pd.read_csv(product_path)
    data = order_data.merge(lineitem_data, left_on='o_order_id', right_on='li_order_id')
    data = data.merge(product_data, left_on='li_product_id', right_on='p_product_id')

    return data[['store', 'department', 'li_order_id', 'date', 'price', 'quantity']]


def pre_process(data: pd.DataFrame) -> pd.DataFrame:
    data['year'] = data['date'].dt.year
    data['week'] = data['date'].dt.week
    data['month'] = data['date'].dt.month
    # reset year in cases where one the last days of the week is in the new year
    # e.g. the last day of week 52 of 2011 is actual 2012-01-01
    # to get the proper year the year has to be reduced by 1 to 2011
    # in general whenever the month of the date is in january but the week is above 50 the year needs to be reduced
    data['year'] = np.where((data['week'] > 50) & (data['month'] == 1), data['year'] - 1, data['year'])
    data['row_price'] = data['quantity'] * data['price']

    grouped = data.groupby(['store', 'department', 'year', 'week'])['row_price'].sum().reset_index()

    def make_date(year, week, weekday):
        date_str = "{}-W{:02d}-{}".format(year, week, weekday)
        date = datetime.datetime.strptime(date_str, '%G-W%V-%u')
        return date

    grouped['date'] = grouped[['week', 'year']].apply(lambda r: make_date(r['year'], r['week'], 5), axis=1)
    grouped = grouped.rename(index=str, columns={'store': 'Store', 'department': 'Dept', 'date': 'Date', 'row_price': 'Weekly_Sales'})
    print("pp done")
    return grouped[['Store', 'Dept', 'Date', 'Weekly_Sales']]

def load_data_in_chunks(order_path: str, lineitem_path: str, product_path: str,
                        chunksize: int = 10000000) -> pd.DataFrame:
    all_chunks = []
    product_data = pd.read_csv(product_path)

    # Initialize iterators for lineitem and order data
    lineitem_iter = pd.read_csv(lineitem_path, chunksize=chunksize)
    order_data = pd.read_csv(order_path, parse_dates=['date'])

    start_order_idx = 0  # To keep track of the starting index for orders

    for lineitem_chunk in lineitem_iter:
        # Determine the range of li_order_id in the current lineitem_chunk
        min_li_order_id = lineitem_chunk['li_order_id'].min()
        max_li_order_id = lineitem_chunk['li_order_id'].max()

        # Find corresponding orders in the order data
        order_chunk = order_data[
            (order_data['o_order_id'] >= min_li_order_id) & (order_data['o_order_id'] <= max_li_order_id)]

        # Merge the chunks
        data = lineitem_chunk.merge(order_chunk, left_on='li_order_id', right_on='o_order_id')
        data = data.merge(product_data, left_on='li_product_id', right_on='p_product_id')

        # Select the required columns
        processed_chunk = data[['store', 'department', 'li_order_id', 'date', 'price', 'quantity']]

        # Append the processed chunk to the list
        all_chunks.append(processed_chunk.reset_index(drop=True))

    final_data = pd.concat(all_chunks, ignore_index=True)
    print("loaddone")
    return final_data


def train(data: pd.DataFrame) -> UseCase03Model:
    """
    Trains an ARIMA or SARIMAX model for each department at each store. The best model for each department is found
    using the `auto_arima` method from `pmdarima`.
    :param data: Pandas DataFrame with columns=[Store, Dept, Date, Weekly_Sales]
    :return: A UseCase03Models
    """
    models = UseCase03Model(use_store=True, use_department=True)
    combinations = np.unique(data[['Store', 'Dept']].apply(lambda r: (r[0], r[1]), axis=1))
    for c in tqdm(combinations, desc='Training'):
        store = c[0]
        dept = c[1]
        ts_data = data[(data.Store == store) & (data.Dept == dept)]
        ts_min = ts_data.Date.min()
        ts_max = ts_data.Date.max()
        ts_data = ts_data.set_index('Date')['Weekly_Sales'].sort_index()
        print(c)
        print(ts_data.shape)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # add freq='W-Fri' would fail
            model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=52).fit()
        print(f"{store},{dept},{ts_min},{ts_max}")
        models.store_model(store, dept, model, ts_min, ts_max)

    return models


def serve(model: UseCase03Model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Create forecasts using the given model for the given deprartments and stores.
    :param model: The trained models for each department and each store
    :param data: Pandas DataFrame containing the stores, departments, and the desired number of periods for the forecast
    :return: The forecasts for each department and each store of the desired length
    """
    # compute forecast for all store/department combinations in the data set
    forecasts = pd.DataFrame(columns=['store', 'department', 'date', 'weekly_sales'])
    # combinations = np.unique(data[['Store', 'Dept']].values, axis=0)
    for index, row in tqdm(data.iterrows(), desc='Forecasting', total=len(data)):
        store = row.store
        dept = row.department
        periods = int(row.periods)
        try:
            current_model, ts_min, ts_max = model.get_model(store, dept)
        except KeyError:
            continue
        # disable warnings that non-date index is returned from forecast
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ValueWarning)
            forecast = current_model.forecast(periods)
            forecast = np.clip(forecast, a_min=0.0, a_max=None)  # replace negative forecasts
        start = pd.date_range(ts_max, periods=2)[1]
        forecast_idx = pd.date_range(start, periods=periods, freq='W-FRI')
        df = pd.DataFrame({'store': store, 'department': dept, 'date': forecast_idx, 'weekly_sales': forecast})
        forecasts = forecasts.append(df)

    return forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("path", nargs='+')

    # configuration parameters
    args = parser.parse_args()
    order_path = args.path[0]

    stage = args.stage
    work_dir = Path(args.workdir)
    if args.output:
        output = Path(args.output)
    else:
        output = work_dir

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output):
        os.makedirs(output)

    # derivative configuration parameters
    model_file = work_dir / 'uc03.python.model'

    if stage == 'training':
        lineitem_path = args.path[1]
        product_path = args.path[2]
        start = timeit.default_timer()
        data = load_data_in_chunks(order_path, lineitem_path, product_path)
        print(data.head())
        end = timeit.default_timer()
        load_time = end - start
        print('load time:\t', load_time)

        start = timeit.default_timer()
        data = pre_process(data)
        end = timeit.default_timer()
        pre_process_time = end - start
        print('pre-process time:\t', pre_process_time)

        start = timeit.default_timer()
        models = train(data)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        joblib.dump(models, model_file)

    elif stage == 'serving':
        start = timeit.default_timer()
        data = pd.read_csv(order_path)
        print(data.head())
        print("s: " + order_path)
        end = timeit.default_timer()
        load_time = end - start
        print('load time:\t', load_time)

        models = joblib.load(model_file)

        start = timeit.default_timer()
        forecasts = serve(models, data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        forecasts.to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
