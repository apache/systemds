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
import collections
import copy
import logging
import math
import os
import re
import shutil
import socket
import sqlite3
import sys
import threading
import time
from hashlib import sha256
# from Crypto.Hash import SHA256
from pathlib import Path
from itertools import groupby
import pdb
import numpy as np
import pandas as pd
import sklearn.metrics
import yaml

from . import metrics
from .data import *
from .data_generation import DataGeneration
from .database import DatabaseQueue
from .logger import FileAndDBLogger, LogDbHandler, LogPerformanceDbHandler
from .subprocess_util import run_and_capture, run_as_daemon, Stream, stop_daemons, state # nosec - see the subprocess_util module for justification

TPCXAI_VERSION = "1.0.3.1"
#Phase.DATA_GENERATION,Phase.SCORING_DATAGEN,
DEFAULT_PHASES = [Phase.CLEAN, Phase.DATA_GENERATION, Phase.SCORING_DATAGEN,Phase.LOADING, Phase.TRAINING, Phase.SERVING, Phase.SERVING_THROUGHPUT,
                  Phase.SCORING_LOADING, Phase.SCORING, Phase.VERIFICATION, Phase.CHECK_INTEGRITY]
#
CHOICES_PHASES = [Phase.CLEAN, Phase.DATA_GENERATION, Phase.SCORING_DATAGEN,Phase.LOADING, Phase.TRAINING, Phase.SERVING,
                  Phase.SCORING_LOADING, Phase.SCORING, Phase.VERIFICATION, Phase.SERVING_THROUGHPUT,
                  Phase.CHECK_INTEGRITY]
#DEFAULT_USE_CASES = [1,11,3,4,6,16,7,17,9,19,10,20]
#CHOICES_USE_CASES = [1,11,3,4,6,16,7,17,9,19,10,20]
DEFAULT_USE_CASES = [1]
CHOICES_USE_CASES = [1]
DEFAULT_CONFIG_PATH = Path('config/default.yaml')

FILE_PATH_REGEX = r'[A-Za-z0-9/\_\-\.]+'
PHASE_PLACEHOLDER = "_PHASE_PLACEHOLDER_"
STREAM_PLACEHOLDER ="_STREAM_PLACEHOLDER_"


def load_configuration(dat_path: Path, config_path: Path):
    """
    Loads all configuration file and merges them, the config files from the dat directory always take precedence
    :param dat_path: Directory containing the config files the must not be change by the average user
    :param config_path: File containing all properties defined by the user
    :return:
    """
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        #config = benedict(config)
    for c in dat_path.glob('*.yaml'):
        with open(c, 'r') as dat_file:
            dat_config = yaml.safe_load(dat_file)
            if len(dat_config.items()) > 0:
                config=merge_dict(config, dat_config)

    return config

# based on python-benedict's merge (https://github.com/fabiocaccamo/python-benedict: MIT License)
def merge_dict(d1, d2):
    for key, value in d2.items():
        merge_item(d1, key, value)
    return d1

# based on python-benedict's merge (https://github.com/fabiocaccamo/python-benedict: MIT License)
def merge_item(d1, key, value):
    if key in d1:
        item = d1.get(key, None)
        if isinstance(item,dict) and isinstance(value,dict):
            merge_dict(item, value)
        elif isinstance(item,list) and isinstance(value,list):
           item += value
        else:
            d1[key] = value
    else:
        d1[key] = value


def get_estimated_sf_for_size(desired_value, config_dir):
    def calculate_size(scaling_factor):
        # number of rows
        customer_size = math.sqrt(scaling_factor) * 100000
        weeks = (math.log(scaling_factor, 10) + 2) * 52
        product_size = math.sqrt(scaling_factor) * 1000
        order_size = customer_size * 0.5 * weeks
        lineitem_size = order_size * 6.5
        order_returns_size = lineitem_size * 0.1 / 2
        financial_customer_size = customer_size * 0.1
        financial_transactions_size = financial_customer_size * weeks * 10
        failure_samples = math.sqrt(scaling_factor) * 100
        disk_size = math.sqrt(scaling_factor) * 1000
        failures_size = failure_samples * disk_size
        marketplace_size = customer_size * 0.1 * 10
        identities = customer_size * 0.0001
        images_per_identity = math.sqrt(scaling_factor) * 10
        images_size = identities * images_per_identity
        ratings_per_customer = math.log(scaling_factor, 10) + 20
        ratings_size = customer_size * ratings_per_customer * 0.1
        conversation_size = math.sqrt(30.0*scaling_factor) * 100

        # size estimates in kB
        customer_kb = customer_size * 0.12
        product_kb = product_size * 0.034
        order_kb = order_size * 0.04
        lineitem_kb = lineitem_size * 0.021
        order_returns_kb = order_returns_size * 0.016
        financial_customer_kb = financial_customer_size * 0.0137
        financial_transactions_kb = financial_transactions_size * 0.078
        failures_kb = failures_size * 0.167
        marketplace_kb = marketplace_size * 0.176
        images_kb = images_size * 160.914
        ratings_kb = ratings_size * 0.013
        conversation_kb = conversation_size * 68.207

        esitmated_size = math.fsum(
            [customer_kb, product_kb, order_kb, lineitem_kb, order_returns_kb, financial_customer_kb,
             financial_transactions_kb, failures_kb, marketplace_kb, images_kb, ratings_kb, conversation_kb])

        return esitmated_size / 1024 / 1024

    init_range = 1e17
    found = False
    current_sf = init_range // 2
    last_three_sf = collections.deque([-1, -1, -1], maxlen=3)
    correction = current_sf // 2
    # save the best sf to prevent oscillation
    tolerance = 1e-5
    i = 0

    while not found:
        i += 1
        current_size = calculate_size(current_sf)
        # terminate if current size is within tolerance
        if abs(current_size - desired_value) / desired_value < tolerance:
            found = True
        # or worse than the best value the has been found the latter means oscillation
        elif current_sf == last_three_sf[-2]:
            found = True
        elif current_size < desired_value:
            last_three_sf.append(current_sf)
            current_sf = current_sf + correction
            correction = correction // 2
            if correction < 0.125:
                correction = 0.125
        else:
            last_three_sf.append(current_sf)
            current_sf = current_sf - correction
            correction = correction // 2
            if correction < 0.125:
                correction = 0.125

    def sf_size_error_tuple(sf, target_value):
        size = calculate_size(sf)
        error = abs(size - desired_value)
        return sf, size, error

    best_lst = list(sorted(map(lambda sf: sf_size_error_tuple(sf, desired_value), last_three_sf), key=lambda t: t[2]))
    best_sf = best_lst[0][0]
    return best_sf


def validate_arg_is_printable(value, regex):
    if value.isprintable() is False or re.match(regex, value) is None:
        print("Argument contains invalid values")
        exit(1)
    return True


def validate_argument(value, regex):
    if type(value) is list:
        for element in value:
            validate_arg_is_printable(str(element), regex)
    elif type(value) is str:
        validate_arg_is_printable(value, regex)
    else:
        print("Argument contains invalid values")
        exit(1)
    return True


def mangle_url(datastore, url):
    if datastore.name == 'local_fs':
        return Path(url)
    else:
        return url


def guess_prediction_label(columns, label_column,
                           common_names=None):
    if common_names is None:
        common_names = ['prediction', 'predictions', 'pred', 'preds', 'forecast', 'forecasts']
    if label_column in columns:
        return label_column
    else:
        # search all columns for a column with a common name used for predictions
        # it is expected that only one such column exists
        column_candidates = [c for c in columns if c in common_names]
        if len(column_candidates) > 0:
            return column_candidates[0]
        else:
            return None


def load_metric(modules, metric_name):
    metric_found = False
    metric = None
    for module in modules:
        try:
            metric = getattr(module, metric_name)
            metric_found = True
        except AttributeError:
            continue

    if metric_found:
        return metric
    else:
        raise AttributeError(f"No {metric_name} found in ")


def scoring(true_labels, pred_labels, label_column, metric_name, labels=None, delimiter=',', sort_predictions=False,
            **kwargs):
    if not metric_name:
        return -1.0
    csv_engine = 'python' if len(delimiter.encode('utf-8')) > 1 else 'c'
    true_labels = pd.read_csv(true_labels, delimiter=delimiter, engine=csv_engine)
    pred_labels = pd.read_csv(pred_labels, delimiter=delimiter, engine=csv_engine)
    prediction_label = guess_prediction_label(pred_labels.columns, label_column)
    if sort_predictions:
        sort_columns = [c for c in pred_labels.columns if c != prediction_label]
        pred_labels = pred_labels.sort_values(by=sort_columns)
    metric_fun = load_metric([metrics, sklearn.metrics], metric_name)

    if labels is not None:
        kwargs['labels'] = labels
    if prediction_label is not None:
        cols_in_common = set(true_labels.columns).intersection(pred_labels.columns)
        join_cols = list(cols_in_common.difference([prediction_label]))
        data = true_labels.merge(pred_labels, on=join_cols)
        metric = metric_fun(data[f"{label_column}_x"], data[f"{prediction_label}_y"], **kwargs)
    else:
        metric = metric_fun(true_labels[label_column], pred_labels, **kwargs)
    return metric


def init_db(database_path):
    database = sqlite3.connect(str(database_path))

    # create schema
    database.execute('''
        CREATE TABLE IF NOT EXISTS benchmark (
            benchmark_sk    INTEGER NOT NULL,   -- UUID of the benchmark 
            version         TEXT,               -- version of the benchmark kit
            hostname        TEXT NOT NULL,      -- hostname of the tpcxai-driver (where the benchmark was started)
            start_time      FLOAT NOT NULL,     -- timestamp when the benchmark was initiated
            end_time        FLOAT,              -- timestamp when the benchmark was stopped/finished
            scale_factor    INT NOT NULL,       -- scale factor for the data generation
            tpcxai_home     TEXT NOT NULL,      -- home directory of the benchmark
            config_path     TEXT NOT NULL,      -- path to the config file
            config          TEXT NOT NULL,      -- content of the config file
            cmd_flags       TEXT NOT NULL,      -- flags that were used to start the benchmark `bin/tpcxai.sh ...`
            benchmark_name  TEXT,               -- user specified name for the benchmark
            successful      INTEGER,            -- was the benchmark successful (all use-cases and phases were run)
            PRIMARY KEY (benchmark_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS command (
            command_sk     INTEGER NOT NULL,
            benchmark_fk    INTEGER NOT NULL,
            use_case        INTEGER NOT NULL,
            phase           TEXT NOT NULL,
            phase_run       INTEGER,
            sub_phase       TEXT NOT NULL,
            command         TEXT NOT NULL,
            start_time      FLOAT,
            end_time        FLOAT,
            runtime         INTEGER,
            return_code     INTEGER,            -- command finished successfully (0) or failed (!= 0)
            PRIMARY KEY (command_sk),
            FOREIGN KEY (benchmark_fk) REFERENCES benchmark(benchmark_sk) 
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS log_std_out (
            use_case_fk     INTEGER NOT NULL,   -- part of key
            part            INT,                -- which part of the log file
            log             TEXT,               -- actual content of this part of the log
            PRIMARY KEY (use_case_fk, part),
            FOREIGN KEY (use_case_fk) REFERENCES command(command_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS log_std_err (
            use_case_fk     INTEGER NOT NULL,   -- part of key
            part            INT,                -- which part of the log file
            log             TEXT,               -- actual content of this part of the log
            PRIMARY KEY (use_case_fk, part),
            FOREIGN KEY (use_case_fk) REFERENCES command(command_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS stream (
            use_case_fk     INTEGER NOT NULL,
            stream          TEXT,
            PRIMARY KEY (use_case_fk, stream),
            FOREIGN KEY (use_case_fk) REFERENCES command(command_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS quality_metric (
            use_case_fk     INTEGER NOT NULL,
            metric_name     TEXT,
            metric_value    FLOAT,
            PRIMARY KEY (use_case_fk, metric_name),
            FOREIGN KEY (use_case_fk) REFERENCES command(command_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS performance_metric (
            benchmark_fk    INTEGER NOT NULL,
            metric_name     TEXT,
            metric_value    FLOAT,
            metric_time     FLOAT,
            PRIMARY KEY (benchmark_fk, metric_name),
            FOREIGN KEY (benchmark_fk) REFERENCES benchmark(benchmark_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS benchmark_files (
            benchmark_fk    INTEGER NOT NULL,   -- benchmark id
            relative_path   TEXT,               -- path to the file, relative to TPCxAI_HOME
            absolute_path   TEXT,               -- absolute path of the file
            sha256          TEXT,               -- sha256 checksum of the file
            PRIMARY KEY (benchmark_fk, relative_path),
            FOREIGN KEY (benchmark_fk) REFERENCES benchmark(benchmark_sk)
        );
    ''')
    database.execute('''
        CREATE TABLE IF NOT EXISTS timeseries (
            benchmark_fk        INTEGER NOT NULL,   -- benchmark id
            hostname            TEXT,               -- hostname where this event occured
            name                TEXT,               -- name of the timeseries
            timestamp           INTEGER,            -- instant (timestamp) of the event
            value               TEXT,               -- value of the event
            unit                TEXT,               -- unit of the timeseries
            PRIMARY KEY (benchmark_fk, hostname, name, timestamp),
            FOREIGN KEY (benchmark_fk) REFERENCES benchmark(benchmark_sk)
        );
    ''')

    database.commit()
    cursor = database.cursor()
    res = cursor.execute(f"SELECT * FROM pragma_table_info('command') WHERE name == 'start_time'")
    is_empty = True
    if res.fetchone():
        is_empty = False
    if is_empty:
        database.execute('ALTER TABLE command ADD COLUMN start_time FLOAT NOT NULL DEFAULT 0.0')
        is_empty = True
    res = cursor.execute(f"SELECT * FROM pragma_table_info('command') WHERE name == 'end_time'")
    if res.fetchone():
        is_empty = False
    if is_empty:
        database.execute('ALTER TABLE command ADD COLUMN end_time FLOAT NOT NULL DEFAULT 0.0')
    res = cursor.execute(f"SELECT * FROM pragma_table_info('command') WHERE name == 'phase_run'")
    if not res.fetchone():
        database.execute('ALTER TABLE command ADD COLUMN phase_run INTEGER')
    res = cursor.execute(f"SELECT * FROM pragma_table_info('benchmark') WHERE name == 'version'")
    if not res.fetchone():
        database.execute('ALTER TABLE benchmark ADD COLUMN version TEXT')
    database.commit()

    return database


def compute_tpcxai_metric(logs,SF,aiucpm_logger,datagen_in_tload=False,num_streams=2):
    TLD = TPTT = TPST1 = TPST2 = TPST = TTT = AICUPM = -1
    TOTAL_USE_CASES = 10

    #DATAGEN
    TDATA_GEN=0.0
    datagen_times=logs[(logs['sub_phase']==SubPhase.WORK) & (logs['phase']==Phase.DATA_GENERATION)][['phase','sub_phase','metric', 'start_time', 'end_time']]
    TDATA_GEN=np.sum(datagen_times[['metric']].values)
    if TDATA_GEN>0:
       aiucpm_logger.info(f'DATAGEN: {round(TDATA_GEN,3)}')

    #TLD - LOADING + Datagen(if included)
    loading_times = logs[(logs['sub_phase'] == SubPhase.WORK) & (logs['phase'] == Phase.LOADING)][['phase', 'sub_phase', 'metric', 'start_time', 'end_time']]
    TLOAD = loading_times['end_time'].max() - loading_times['start_time'].min()
    if TLOAD>0:
        aiucpm_logger.info(f'TLOAD: {round(TLOAD,3)}')

    if datagen_in_tload == True and (TLOAD+TDATA_GEN)>0:
        TLOAD += TDATA_GEN
        aiucpm_logger.info(f'TLOAD(TLOAD+TDATA_GEN): {round(TLOAD,3)}')


    TLD=1.0*TLOAD
    if TLOAD>0:
        aiucpm_logger.info(f'TLD: {round(TLD,3)}')

    #TPTT - TRAINING
    training_times=logs[(logs['sub_phase']==SubPhase.WORK) & (logs['phase']==Phase.TRAINING)][['metric']].values
    if len(training_times) == TOTAL_USE_CASES:
        TPTT=(np.prod(training_times)**(1.0/len(training_times)))
        if TPTT>0:
            aiucpm_logger.info(f'TPTT: {round(TPTT,3)}')

    #TPST1 - SERVING1
    serving_times1=logs[(logs['sub_phase']==SubPhase.WORK) & (logs['phase']==Phase.SERVING) & (logs['phase_run']==1)][['metric']].values
    if len(serving_times1) == TOTAL_USE_CASES:
        TPST1=(np.prod(serving_times1)**(1.0/len(serving_times1)))
        if TPST1>0:
            aiucpm_logger.info(f'TPST1: {round(TPST1,3)}')

    #TPST2 - SERVING2
    serving_times2=logs[(logs['sub_phase']==SubPhase.WORK) & (logs['phase']==Phase.SERVING) & (logs['phase_run']==2)][['metric']].values
    if len(serving_times2) == TOTAL_USE_CASES:
        TPST2=(np.prod(serving_times2)**(1.0/len(serving_times2)))
        if TPST2>0:
            aiucpm_logger.info(f'TPST2: {round(TPST2,3)}')

    #TPST
    TPST=max(TPST1,TPST2)
    if TPST>0:
        aiucpm_logger.info(f'TPST: {round(TPST,3)}')

    #THROUGHPUT
    throughput_results = logs[(logs['sub_phase'] == SubPhase.WORK) & (logs['phase'] == Phase.SERVING_THROUGHPUT)]
    max_throughput_results = throughput_results['end_time'].max() - throughput_results['start_time'].min()
    throughput_results = throughput_results.groupby(['stream'])['metric'].sum()
    TTT=-1
    if len(throughput_results) == num_streams:
        TTT = max_throughput_results / (len(throughput_results) * TOTAL_USE_CASES)
        if TTT > 0:
            aiucpm_logger.info(f'TTT: {round(TTT, 3)}')

    #AIUCPM
    if datagen_in_tload==True and TDATA_GEN<=0:
       return -1

    if TLD>0 and TPTT>0 and TPST>0 and TTT>0 and SF>0:
        AIUCPM_numerator=60.0*SF*TOTAL_USE_CASES
        AIUCPM_denominator=(TLD*TPTT*TPST*TTT)**(1.0/4.0)
        if AIUCPM_denominator>0 and AIUCPM_numerator>0:
            AIUCPM = 1.0*(AIUCPM_numerator/AIUCPM_denominator)
            return round(AIUCPM,3)

    return -1


def merge_actions(actions: List[Action], phase_to_merge: Phase) -> List[Action]:
    filtered_actions = list(filter(lambda a: a.phase.value == phase_to_merge.value, actions))
    unified_actions = {}
    for action in filtered_actions:
        if isinstance(action, SubprocessAction) and action.command not in unified_actions:
            new_action = copy.deepcopy(action)
            new_action.use_case = 0
            unified_actions[action.command] = new_action

    new_actions = [copy.deepcopy(a) for a in actions if a not in filtered_actions]
    new_actions.extend(unified_actions.values())
    return new_actions


def hash_file(file: Path, buffer_size=64*1024):
    if not file.exists():
        print(f"{file} does not exist")
        return '-1'
    done = False
    checksum = sha256()
    with open(file, 'rb', buffering=0) as bin_file:
        while not done:
            data = bin_file.read(buffer_size)
            if len(data) > 0:
                checksum.update(data)
            else:
                done = True

    return checksum.hexdigest()


def main():
    # driver_home
    driver_home = Path('.').resolve()

    # adabench_home
    tpcxai_home = Path("..").resolve()

    # adabench data generator
    datagen_home = tpcxai_home / 'data-gen'
    datagen_config = datagen_home / 'config'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, default=DEFAULT_CONFIG_PATH)
    parser.add_argument('--dat', required=False, default=Path('dat'))
    parser.add_argument('--phase', required=False, nargs='*', default=DEFAULT_PHASES,
                        choices=list(map(lambda p: p.name, CHOICES_PHASES)))
    parser.add_argument('-sf', '--scale-factor', metavar='scale_factor', required=False, default=1, type=float)
    parser.add_argument('-uc', '--use-case', metavar='use_case', type=int, required=False, nargs='*',
                        default=DEFAULT_USE_CASES, choices=CHOICES_USE_CASES)
    parser.add_argument('-eo', '--execution-order', metavar='order', required=False, default='phase',
                        choices=['phase', 'use-case'])
    parser.add_argument('--ttvf', required=False, metavar='factor', default=0.1, type=float)
    parser.add_argument('--streams', required=False, metavar='N', default=2, type=int)
    #parser.add_argument('--data-gen', required=False, metavar='data_gen', default=False, type=bool)
    parser.add_argument('--data-gen', required=False, action='store_true', help='Enable data generation')
    # flags
    parser.add_argument('-v', '--verbose', required=False, default=False, action='store_true')
    parser.add_argument('--dry-run', required=False, default=False, action='store_true')
    parser.add_argument('--no-unified-load', required=False, default=False, action='store_true')

    args = parser.parse_args()

    # validate arguments
    # validate_argument(args.config, FILE_PATH_REGEX)

    # handle arguments
    config_path = Path(args.config) if isinstance(args.config, str) else args.config
    dat_path = Path(args.dat) if isinstance(args.dat, str) else args.dat
    phases = list(map(lambda p: Phase[p] if type(p) is str else p, args.phase))
    if Phase.SCORING in phases and Phase.SCORING_DATAGEN not in phases:
        phases.append(Phase.SCORING_DATAGEN)
        phases.append(Phase.SCORING_LOADING)
    data_gen = args.data_gen
    print(f"data_gen: {data_gen}")

    # add clean before data generation
    if Phase.DATA_GENERATION in phases and Phase.CLEAN not in phases:
        phases.append(Phase.CLEAN)

    # phases = args.phase
    scale_factor = args.scale_factor
    internal_scale_factor = get_estimated_sf_for_size(args.scale_factor, driver_home / 'config')
    ttvf = args.ttvf
    use_cases = args.use_case
    execution_order = args.execution_order
    num_streams = args.streams

    # handle flags
    verbose = args.verbose
    dry_run = args.dry_run
    no_unified_load = args.no_unified_load

    config = load_configuration(dat_path, config_path)

    # FIXME check if default.yaml is loaded and replace all slashes in URLs with backslashes if driver is run on Windows

    workload = config['workload']
    engine_base = workload.get('engine_base','')
    defaul_delimiter = workload['delimiter']

    streams_mapping_keys = [x for x in workload.keys() if x.startswith('serving_throughput_stream')]
    if num_streams < 2:
        print(f'number of desired streams ({num_streams}) has to be at least 2')
        exit(1)
    if num_streams > len(streams_mapping_keys):
        print(f'number of desired streams ({num_streams}) exceeds the number of available streams ({len(streams_mapping_keys)})')
        exit(1)
    streams_mapping_keys = streams_mapping_keys[:num_streams]
    streams_mapping = [workload[s] for s in streams_mapping_keys]
    usecases_config = workload['usecases']

    pdgf_home = Path(workload['pdgf_home'])
    if not pdgf_home.is_absolute():
        pdgf_home = tpcxai_home / pdgf_home
    pdgf_java_opts = os.getenv('TPCxAI_PDGF_JAVA_OPTS', '')
    datagen_output = Path(workload['raw_data_url'])
    if not datagen_output.is_absolute():
        # if the url is relative make it relative to tpcxai_home
        datagen_output = tpcxai_home / datagen_output

    temp_dir = Path(workload['temp_dir'])

    timeseries = [DaemonAction(0, ts, Phase.INIT, SubPhase.WORK, working_dir=tpcxai_home) for ts in workload['timeseries']]

    # print benchmark info
    print(f"tpcxai_home: {tpcxai_home}")

    is_datagen_parallel = bool(workload['pdgf_node_parallel'])
    if data_gen:
        print("flas works")
        datagen = DataGeneration(1234, tpcxai_home, pdgf_home, pdgf_java_opts, datagen_home, datagen_config, datagen_output,
                                scale_factor=internal_scale_factor, ttvf=ttvf, parallel=is_datagen_parallel)
    # make sure that the scoring data is locally available for the tpcxai-driver
    # this is only necessary if the node parallel data gen is used,
    # since it's not guranteed to be run on the tpcxai-driver node.
    # the local scoring data resides in the specified temp directory, `/tmp` by default

        if is_datagen_parallel:
            datagen_output_local = temp_dir / 'scoring'
            pdgf_home_local = tpcxai_home / 'lib' / 'pdgf'
            if not datagen_output_local.is_absolute():
                # if the url is relative make it relative to tpcxai_home
                datagen_output_local = tpcxai_home / datagen_output_local
            datagen_local = DataGeneration(1234, tpcxai_home, pdgf_home_local, pdgf_java_opts, datagen_home, datagen_config, datagen_output_local,
                                       scale_factor=internal_scale_factor, ttvf=ttvf, parallel=False)
    

    actions = []
    collected_tables = set()
    loaded_files_set = set()

    # read the configuration for each use case
    # create the appropriate actions for all phases and sub-phases
    # relevant actions are filtered later, when creating an execution plan
    for uc_key in use_cases:
        uc = usecases_config[uc_key]
        name = uc['name']
        tables = uc['tables']
        raw_files = uc['raw_data_files']
        if 'raw_data_folder' in uc:
            raw_folder = uc['raw_data_folder']
        else:
            raw_folder = None
        label_column = uc['label_column']
        if 'delimiter' in uc:
            delimiter = uc['delimiter']
        else:
            delimiter = defaul_delimiter
        if 'labels' in uc.keys():
            labels = uc['labels']
        else:
            labels = None

        # quality metrics
        if 'scoring_sort' in uc.keys():
            scoring_sort = uc['scoring_sort']
        else:
            scoring_sort = False
        scoring_metric = uc['quality_metric']
        scoring_kvargs = uc['quality_metric_kvargs'] if 'quality_metrics_kvargs' in uc.keys() else {}
        quality_metric_threshold = uc['quality_metric_threshold']
        quality_metric_larger_is_better = uc['quality_metric_larger_is_better']

        # engines
        training_engine = Template(uc['training_engine']).substitute(tpcxai_home=tpcxai_home, engine_base=engine_base)
        serving_engine = Template(uc['serving_engine']).substitute(tpcxai_home=tpcxai_home, engine_base=engine_base)

        # data stores
        datagen_datastore = datastore_from_dict(workload['datagen_datastore'])
        training_datastore = datastore_from_dict(uc['training_datastore'])
        serving_datastore = datastore_from_dict(uc['serving_datastore'])
        model_datastore = datastore_from_dict(uc['model_datastore'])
        output_datastore = datastore_from_dict(uc['output_datastore'])

        # templates
        training_template = Template(uc['training_template'])
        serving_template = Template(uc['serving_template'])
        serving_throughput_template = Template(uc['serving_throughput_template'])
        if 'postwork_training_template' in uc:
            postwork_training_template = Template(uc['postwork_training_template'])
        else:
            postwork_training_template = None

        # URLs
        training_data_url = mangle_url(training_datastore, uc['training_data_url'])
        serving_data_url = mangle_url(serving_datastore, uc['serving_data_url'])
        scoring_data_url = mangle_url(serving_datastore, uc['scoring_data_url'])
        model_url = mangle_url(model_datastore, uc['model_url'])
        output_url = mangle_url(output_datastore, uc['output_url'])
        scoring_output_url = mangle_url(output_datastore, uc['scoring_output_url'])
        working_dir = None
        if 'working_dir' in uc:
            working_dir = uc['working_dir']

        # generate data for this use case, i.e. add relevant tables
        collected_tables.update(tables)
        if data_gen:
            # clean up generated data <<---
            for raw_file in raw_files:
                file_path = datagen_output / 'training' / raw_file
                rm_training_data_gen = datagen_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_training_data_gen, Phase.CLEAN, SubPhase.WORK))

                file_path = datagen_output / 'serving' / raw_file
                rm_serving_data_gen = datagen_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_serving_data_gen, Phase.CLEAN, SubPhase.WORK))

                file_path = datagen_output / 'scoring' / raw_file
                rm_scoring_data_gen = datagen_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_scoring_data_gen, Phase.CLEAN, SubPhase.WORK))

            if raw_folder:
                folder_path = datagen_output / 'training' / raw_folder
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_training_data_gen = datagen_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_training_data_gen, Phase.CLEAN, SubPhase.WORK))

                folder_path = datagen_output / 'serving' / raw_folder
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_serving_data_gen = datagen_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_serving_data_gen, Phase.CLEAN, SubPhase.WORK))

                folder_path = datagen_output / 'scoring' / raw_folder
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_scoring_data_gen = datagen_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_scoring_data_gen, Phase.CLEAN, SubPhase.WORK))

            # clean up parallel-generated data in other nodes
            if is_datagen_parallel:
                if datagen_datastore.delete_parallel:
                    folder_path = datagen_output
                    rm_parallel_data_gen = datagen_datastore.delete_parallel.substitute(destination=folder_path)
                    #print(rm_parallel_data_gen)
                    actions.append(SubprocessAction(uc_key, rm_parallel_data_gen, Phase.CLEAN, SubPhase.WORK))
                else:
                    print('The delete_parallel configuration for the datagen_datastore object does not exist.',file=sys.stderr)
                    sys.exit(1)

            # clean loaded data
            for raw_file in raw_files:
                file_path = f"{training_data_url}/{raw_file}"
                rm_training_data = training_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_training_data, Phase.CLEAN, SubPhase.WORK))

                file_path = f"{serving_data_url}/{raw_file}"
                rm_serving_data = training_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_serving_data, Phase.CLEAN, SubPhase.WORK))

                file_path = f"{scoring_data_url}/{raw_file}"
                rm_scoring_data = training_datastore.delete.substitute(destination=file_path)
                actions.append(SubprocessAction(uc_key, rm_scoring_data, Phase.CLEAN, SubPhase.WORK))

            if raw_folder:
                folder_path = f"{training_data_url}/{raw_folder}"
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_training_data = training_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_training_data, Phase.CLEAN, SubPhase.WORK))

                folder_path = f"{serving_data_url}/{raw_folder}"
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_serving_data = training_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_serving_data, Phase.CLEAN, SubPhase.WORK))

                folder_path = f"{scoring_data_url}/{raw_folder}"
                folder_path = f"{folder_path} {folder_path}.seq"
                rm_scoring_data = training_datastore.delete.substitute(destination=folder_path)
                actions.append(SubprocessAction(uc_key, rm_scoring_data, Phase.CLEAN, SubPhase.WORK))

            # clean scoring labels
            raw_file_name, raw_file_extension = str.split(raw_files[0], '.')[0:2]
            data_dir = datagen_output / 'scoring' #if not is_datagen_parallel #else datagen_output_local / 'scoring'
            file_path = data_dir / (raw_file_name + '_labels.' + raw_file_extension)
            rm_scoring_data = training_datastore.delete.substitute(destination=file_path)
            actions.append(SubprocessAction(uc_key, rm_scoring_data, Phase.CLEAN, SubPhase.WORK))

            # clean models and predictions
            rm_model = training_datastore.delete.substitute(destination=model_url)
            actions.append(SubprocessAction(uc_key, rm_model, Phase.CLEAN, SubPhase.WORK))
            rm_output = output_datastore.delete.substitute(destination=output_url)
            actions.append(SubprocessAction(uc_key, rm_output, Phase.CLEAN, SubPhase.WORK))



        # load training and serving data
        create_training_cmd = training_datastore.create.substitute(destination=training_data_url)
        actions.append(SubprocessAction(uc_key, create_training_cmd, Phase.LOADING, SubPhase.PREPARATION))

        raw_files_set = set()
        for raw_file in raw_files:
            if datagen_output / 'training' / raw_file not in loaded_files_set:
                raw_files_set.add(datagen_output / 'training' / raw_file)
        if raw_folder:
            if datagen_output / 'training' / raw_folder not in loaded_files_set:
                raw_files_set.add(datagen_output / 'training' / raw_folder)

        raw_files_as_posix_str=' '.join([x.as_posix() for x in list(raw_files_set)])
        load_training_cmd = training_datastore.load.substitute(destination=training_data_url,source=raw_files_as_posix_str)
        actions.append(SubprocessAction(uc_key, load_training_cmd, Phase.LOADING, SubPhase.WORK))
        loaded_files_set.update(raw_files_set)

        create_serving_cmd = serving_datastore.create.substitute(destination=serving_data_url)
        actions.append(SubprocessAction(uc_key, create_serving_cmd, Phase.LOADING, SubPhase.PREPARATION))

        raw_files_set.clear()
        for raw_file in raw_files:
            if datagen_output / 'serving' / raw_file not in loaded_files_set:
                raw_files_set.add(datagen_output / 'serving' / raw_file)
        if raw_folder:
            if datagen_output / 'serving' / raw_folder not in loaded_files_set:
                raw_files_set.add(datagen_output / 'serving' / raw_folder)
        raw_files_as_posix_str = ' '.join([x.as_posix() for x in list(raw_files_set)])
        load_serving_cmd = serving_datastore.load.substitute(destination=serving_data_url,source=raw_files_as_posix_str)
        actions.append(SubprocessAction(uc_key, load_serving_cmd, Phase.LOADING, SubPhase.WORK))
        loaded_files_set.update(raw_files_set)

        # training phase
        training_store_cmd = model_datastore.create.substitute(destination=model_url)
        actions.append(SubprocessAction(uc_key, training_store_cmd, Phase.TRAINING, SubPhase.PREPARATION))
        training_cmd = training_template.substitute(training_engine=training_engine, tpcxai_home=tpcxai_home, name=name,
                                                    engine=training_engine,
                                                    input=training_data_url, file=raw_files[0], output=model_url)
        actions.append(SubprocessAction(uc_key, training_cmd, Phase.TRAINING, SubPhase.WORK, working_dir=working_dir))
        if postwork_training_template is not None:
            postwork_training_cmd = postwork_training_template.substitute(tpcxai_home=tpcxai_home, name=name, output=model_url)
            actions.append(SubprocessAction(uc_key, postwork_training_cmd, Phase.TRAINING, SubPhase.POSTWORK, working_dir=working_dir))

        # serving phase
        serving_store_cmd = output_datastore.create.substitute(destination=output_url)
        actions.append(SubprocessAction(uc_key, serving_store_cmd, Phase.SERVING, SubPhase.PREPARATION))
        serving_cmd = serving_template.substitute(serving_engine=serving_engine, tpcxai_home=tpcxai_home, name=name,
                                                  engine=serving_engine,
                                                  input=serving_data_url, file=raw_files[0], model=model_url,
                                                  output=output_url, phase=PHASE_PLACEHOLDER)
        actions.append(SubprocessAction(uc_key, serving_cmd, Phase.SERVING, SubPhase.WORK, working_dir=working_dir))

        # serving throughput phase
        serving_throughput_store_cmd = output_datastore.create.substitute(destination=output_url)
        actions.append(SubprocessAction(uc_key, serving_throughput_store_cmd, Phase.SERVING_THROUGHPUT, SubPhase.PREPARATION))
        serving_throughput_cmd = serving_throughput_template.substitute(
            serving_engine=serving_engine, tpcxai_home=tpcxai_home, name=name,
            engine=serving_engine,
            input=serving_data_url, file=raw_files[0], model=model_url, output=output_url,
            stream=STREAM_PLACEHOLDER, phase=PHASE_PLACEHOLDER
        )
        actions.append(SubprocessAction(uc_key, serving_throughput_cmd, Phase.SERVING_THROUGHPUT, SubPhase.WORK, working_dir=working_dir))

        # scoring
        # loading for scoring
        create_scoring_cmd = serving_datastore.create.substitute(destination=scoring_data_url)
        actions.append(SubprocessAction(uc_key, create_scoring_cmd, Phase.SCORING_LOADING, SubPhase.PREPARATION))

        raw_files_set.clear()
        datagen_output_scoring = datagen_output
        for raw_file in raw_files:
            if datagen_output_scoring / 'scoring' / raw_file not in loaded_files_set:
                raw_files_set.add(datagen_output_scoring / 'scoring' / raw_file)
        if raw_folder:
            if datagen_output_scoring / 'scoring' / raw_folder not in loaded_files_set:
                raw_files_set.add(datagen_output_scoring / 'scoring' / raw_folder)
        raw_files_as_posix_str = ' '.join([x.as_posix() for x in list(raw_files_set)])
        load_scoring_cmd = serving_datastore.load.substitute(source=raw_files_as_posix_str, destination=scoring_data_url)
        actions.append(SubprocessAction(uc_key, load_scoring_cmd, Phase.SCORING_LOADING, SubPhase.WORK))
        loaded_files_set.update(raw_files_set)

        # serving for scoring
        scoring_store_cmd = output_datastore.create.substitute(destination=output_url)
        actions.append(SubprocessAction(uc_key, scoring_store_cmd, Phase.SCORING, SubPhase.INIT))
        scoring_serving_cmd = serving_template.substitute(serving_engine=serving_engine, tpcxai_home=tpcxai_home, name=name,
                                                          engine=serving_engine,
                                                          input=scoring_data_url, file=raw_files[0], model=model_url,
                                                          output=output_url, phase=PHASE_PLACEHOLDER)
        actions.append(SubprocessAction(uc_key, scoring_serving_cmd, Phase.SCORING, SubPhase.INIT, working_dir=working_dir))

        # copy serving output to local filesystem (download)
        mod_path = tpcxai_home / Path(output_url)
        mod_path.mkdir(exist_ok=True, parents=True)
        scoring_source = str(output_url) + '/' + PHASE_PLACEHOLDER + '/predictions.csv'
        scoring_download_cmd = serving_datastore.download.safe_substitute(source=scoring_source, destination=output_url)
        actions.append(SubprocessAction(uc_key, scoring_download_cmd, Phase.SCORING, SubPhase.PREPARATION))

        # calculate the metric
        raw_file_name, raw_file_extension = str.split(raw_files[0], '.')[0:2]
        data_dir = datagen_output / 'scoring' #if not is_datagen_parallel else datagen_output_local / 'scoring' <---
        true_labels = data_dir / (raw_file_name + '_labels.' + raw_file_extension)
        pred_labels = tpcxai_home / output_url / 'predictions.csv'
        scoring_params = {'true_labels': true_labels, 'pred_labels': pred_labels, 'label_column': label_column,
                          'metric_name': scoring_metric, 'labels': labels, 'delimiter': delimiter,
                          'sort_predictions': scoring_sort, **scoring_kvargs}
        scoring_cmd = ScoringAction(uc_key, scoring_params, Phase.SCORING, SubPhase.WORK)
        # verification
        if Phase.SCORING in phases:
            scoring_cmd.add_verification(scoring_metric, quality_metric_threshold, quality_metric_larger_is_better)
        actions.append(scoring_cmd)
    if data_gen:
        # add data generation actions <---
        action_datagen_train = datagen.run(Phase.TRAINING, collected_tables)
        actions.append(action_datagen_train)
        action_datagen_serve = datagen.run(Phase.SERVING, collected_tables)
        actions.append(action_datagen_serve)
        action_datagen_scoring = datagen.run(Phase.SCORING, collected_tables)
        actions.append(action_datagen_scoring)
        if is_datagen_parallel:
            action_datagen_scoring_local = datagen_local.run(Phase.SCORING, collected_tables)
            actions.append(action_datagen_scoring_local)

    # unified load
    # just keep a single load command per file
    # otherwise use-cases with same data would individually load, i.e. overwrite, the data
    if not no_unified_load:
        actions = merge_actions(actions, Phase.LOADING)

    # unify CLEAN
    actions = merge_actions(actions, Phase.CLEAN)

    # filter actions
    # only keep actions for the phases that were specified
    actions = list(filter(lambda a: a.phase in phases, actions))
    # duplicate phases if they were specified multiple times
    tmp_actions = []
    # rename duplicate phases
    # e.g. TRAINING, TRAINING => TRAINING_1, TRAINING_2
    duplicate_phases = set([x for x in phases if phases.count(x) > 1])
    phase_counter = {}
    for phase in phases:
        if phase in phase_counter:
            phase_counter[phase] += 1
        else:
            phase_counter[phase] = int(1)
        phase_num = phase_counter[phase]
        acs = list(filter(lambda a: a.phase == phase, actions))
        renamed_acs = []
        for a in acs:
            a = copy.deepcopy(a)
            a.run = phase_num
            renamed_acs.append(a)
        tmp_actions.extend(renamed_acs)
    actions = tmp_actions

    stream_names = []
    ucs = []
    phases_col = []
    phases_run_col = []
    sub_phases = []
    commands = []
    start_times = []
    end_times = []
    timings = []
    std_logs = []
    err_logs = []
    qualities = {}

    # TODO define order of execution / run rules here
    # TODO for now execute one stage after another, e.g.
    # generating (uc1, uc2, ..., ucn)
    # loading (uc1, uc2, ..., ucn)
    # training (uc1, uc2, ..., ucn)
    # serving (uc1, uc2, ..., ucn)
    if execution_order == 'phase':
        actions_in_order = []
        for key, group in groupby(actions, key=lambda a: a.use_case):
            actions_in_order.extend(list(group))
        #actions_in_order = list(sorted(actions, key=lambda a: (a.phase.value[1], a.phase.value[0], a.run, a.subphase.value, a.use_case)))
    elif execution_order == 'use-case':
        actions_in_order = []
        for key, group in groupby(actions, key=lambda a: a.use_case):
            actions_in_order.extend(list(group))
        #actions_in_order = list(sorted(actions, key=lambda a: (a.use_case, a.phase.value[1], a.run, a.subphase.value)))
    else:
        actions_in_order = []
        exit(1)
    if data_gen:
        if Phase.DATA_GENERATION in phases or Phase.SCORING in phases:
            actions_in_order.insert(0, datagen.prepare())
         #extract all actions related to SERVING_THROUGHPUT
         #keep THROUGHPUT and non-THROUGHPUT actions separate
    actions_in_order_serving_throuhgput = list(filter(lambda a: a.phase.value == Phase.SERVING_THROUGHPUT.value, actions_in_order))
    actions_in_order = list(filter(lambda a: a.phase.value != Phase.SERVING_THROUGHPUT.value, actions_in_order))

    # prepare logging
    log_dir = tpcxai_home / 'logs'
    if not log_dir.exists():
        log_dir.mkdir()

    current_time = time.localtime()
    current_timestamp = time.time()
    log_suffix = time.strftime('%Y%m%d-%H%M', current_time)
    log_file = log_dir / f"tpcxai-sf{scale_factor}-{log_suffix}.csv"
    i = 1
    while log_file.exists():
        log_file = log_dir / f"tpcxai-sf{scale_factor}-{log_suffix}-{i}.csv"
        i += 1

    # setup and initialize database
    benchmark_sk = None
    db_cursor = None
    if not dry_run:
        database_path = log_dir / 'tpcxai.db'
        database = init_db(database_path)

        # log benchmark meta data
        hostname = socket.gethostname()
        db_cursor = database.cursor()
        db_cursor.execute(
            '''
            INSERT INTO benchmark (hostname, start_time, scale_factor, tpcxai_home, config_path, config, cmd_flags, benchmark_name, version)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (hostname, current_timestamp, scale_factor, str(tpcxai_home), str(config_path), yaml.dump(config), ' '.join(sys.argv), '', TPCXAI_VERSION)
        )
        benchmark_sk = db_cursor.lastrowid
        database.commit()
        # close the current connection
        # all future writes should happen through the DatabaseQueue
        database.close()
        db_queue = DatabaseQueue(database_path)

        # start the resource stat collection
        logger = LogPerformanceDbHandler(benchmark_sk, db_queue)
        daemon_stop_event = threading.Event()

        daemon_threads = [
            run_as_daemon(action.command, logger, daemon_stop_event, cwd=action.working_dir)
            for action in timeseries
        ]
        for thread in daemon_threads:
            thread.start()

    # run the benchmark
    if Phase.CLEAN in phases and not dry_run:
        # remove the temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    if Phase.CHECK_INTEGRITY in phases and not dry_run:
        files_changed = 0
        hashes_file = Path(tpcxai_home, 'driver', 'dat', 'tpcxai.sha256')
        if not hashes_file.exists():
            print(f"Hashes file {hashes_file} was not found")
            stop(daemon_stop_event, db_queue)
            exit(2)
        with open(hashes_file, 'r') as f:
            for line in f:
                parts = line.rstrip().split(', ')
                if len(parts) == 2:
                    file = parts[0]
                    checksum = parts[1]
                    check_file_path = Path(tpcxai_home, file)
                    #if check_file_path.exists()==False:
                    #    print(f"{check_file_path} does not exist")
                    file_hash = hash_file(check_file_path)
                    if file_hash != checksum:
                        print(f"{file} was changed")
                        files_changed += 1
                    # add if file was already logged to db
                    # if not log it with its relative path, absolute path, and sha256 hash
                    res = db_queue.query(
                        "SELECT COUNT(1) FROM benchmark_files WHERE benchmark_fk = ? AND relative_path = ?",
                        (benchmark_sk, file)
                    )
                    if res.fetchone()[0] == 0:
                        db_queue.insert("INSERT INTO benchmark_files VALUES (?, ?, ?, ?)",
                                        (benchmark_sk, file, check_file_path.as_posix(), file_hash))
        if files_changed > 0:
            print(f"{files_changed} files have been changed")
            time.sleep(2)
            #stop(daemon_stop_event, db_queue)
            #exit(2)

    threshold_checks = list()
    for action in actions_in_order:
        phase_run = action.run
        phase_run_str = phase_run
        phase_name = f"{str(action.phase).replace('Phase.', '')}_{phase_run}"
        if not phase_run_str:
            phase_run_str = ''
        else:
            phase_run_str = f'({phase_run_str})'
        if action.subphase.value == SubPhase.INIT.value:
            print(f"initializing {action.phase} {phase_run_str} for uc {action.use_case}")
        elif action.subphase.value == SubPhase.PREPARATION.value:
            print(f"preparing {action.phase} {phase_run_str} for uc {action.use_case}")
        elif action.subphase.value == SubPhase.CLEANUP.value:
            print(f"cleaning up {action.phase} {phase_run_str} for uc {action.use_case}")
        else:
            print(f"running {action.phase} {phase_run_str} for uc {action.use_case}")
        try:
            # get the command or scoring parameters as command
            command = ''
            if isinstance(action, SubprocessAction):
                command = action.command
                command = command.replace(PHASE_PLACEHOLDER, f"{phase_name}")
                action.command = command
            elif isinstance(action, ScoringAction):
                command = action.scoring_params

            if dry_run:
                print(command)
            else:
                query_last_use_case_sk = "SELECT max(command_sk) FROM command"
                use_case_sk = db_queue.query(query_last_use_case_sk).fetchone()[0]
                use_case_sk = use_case_sk + 1 if use_case_sk else 1
                query_command = '''
                    INSERT INTO command (command_sk, benchmark_fk, use_case, phase, phase_run, sub_phase, command)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    '''
                query_command_params = (use_case_sk, benchmark_sk, action.use_case, str(action.phase), action.run, str(action.subphase), str(command))
                db_queue.insert(query_command, query_command_params)
                db_queue.insert('INSERT INTO stream (use_case_fk, stream) VALUES (?, ?)', (use_case_sk, 'POWER_TEST'))
                action_log_file = log_dir / f"{log_file.stem}-{action.phase}-{action.run}-{action.use_case}.out"
                if verbose or action.phase.value == Phase.CLEAN.value:
                    print(command)
                if action.phase.value == Phase.DATA_GENERATION.value or action.phase.value == Phase.SCORING_DATAGEN.value:
                    uses_parallel_data_gen = 'parallel-data-gen.sh' in action.command
                    current_wd = datagen_home if not uses_parallel_data_gen else tpcxai_home
                    if action.subphase.value == SubPhase.INIT.value:
                        with FileAndDBLogger(use_case_sk, action_log_file, db_queue) as logger:
                            run_and_capture(action.command.split(), logger, verbose=True, cwd=current_wd)
                else:
                    if isinstance(action, SubprocessAction) and action.working_dir:
                        current_wd = action.working_dir
                    else:
                        current_wd = tpcxai_home

                start = time.perf_counter()
                start_time = time.time()
                if isinstance(action, ScoringAction):
                    metric_name = action.scoring_params['metric_name']
                    print(metric_name)
                    result = scoring(**action.scoring_params)
                    metric_msg = f"Metric ({metric_name}): {result}"
                    print(metric_msg)
                    db_queue.insert('INSERT INTO quality_metric VALUES (?, ?, ?)', (use_case_sk, metric_name, result))
                    if action.verification:
                        st = time.time()
                        s = time.perf_counter()
                        a_ver = action.verification
                        a_ver.add_metric(result)
                        meets_quality_threshold = a_ver.meets_quality_threshold()
                        e = time.perf_counter()
                        et = time.time()
                        rt = e - s
                        rc = 0 if meets_quality_threshold else 1
                        threshold_checks.append(rc)
                        verification_usecase_sk = use_case_sk + 1
                        qry = '''INSERT INTO command (command_sk, benchmark_fk, use_case, phase, phase_run, sub_phase, command,
                                                      start_time, end_time, runtime, return_code)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
                        qry_params = (verification_usecase_sk, benchmark_sk, a_ver.use_case, str(a_ver.phase), 1,
                                      str(a_ver.subphase), a_ver.get_metric_command(), st, et, rt, rc)
                        db_queue.insert(qry, qry_params)
                        db_queue.insert('INSERT INTO stream (use_case_fk, stream) VALUES (?, ?)',
                                      (verification_usecase_sk, 'POWER_TEST'))
                        print(f"running {a_ver.phase} for uc {a_ver.use_case}")
                        if meets_quality_threshold:
                            print(f"checking threshold: {a_ver.get_metric_command()} OK")
                        else:
                            print(f"checking threshold: {a_ver.get_metric_command()} FAILURE")
                    # set return code to 0 if scoring was successful
                    db_queue.insert('UPDATE command SET return_code = ? WHERE command_sk = ?', (0, use_case_sk))
                    std_logs.append(metric_msg)
                    err_logs.append('')
                    qualities[action.use_case] = (metric_name, result)
                else:
                    with FileAndDBLogger(use_case_sk, action_log_file, db_queue) as logger:
                        result = run_and_capture(action.command, logger, verbose=verbose, cwd=current_wd, shell=True)
                end = time.perf_counter()
                end_time = time.time()
                duration = round(end - start, 3)
                action.duration = duration
                db_queue.insert('''
                    UPDATE command
                    SET start_time = ?, end_time = ?, runtime = ?
                    WHERE command_sk = ?''', (start_time, end_time, duration, use_case_sk))
                print(f"time: {duration}s")

                stream_names.append('POWER_TEST')
                ucs.append(action.use_case)
                phases_col.append(action.phase)
                phases_run_col.append(phase_run)
                sub_phases.append(action.subphase)
                if isinstance(action, SubprocessAction):
                    commands.append(action.command)
                else:
                    commands.append(action.scoring_params)
                start_times.append(start_time)
                end_times.append(end_time)
                timings.append(duration)
                if not isinstance(action, PythonAction) and not isinstance(action, DaemonAction):
                    err_logs.append(result.stderr)
                    std_logs.append(result.stdout)
                    db_queue.insert('UPDATE command SET return_code = ? WHERE command_sk = ?',
                                    (result.returncode, use_case_sk))

                    if result.returncode == 0:
                        print('SUCCESS')
                    else:
                        print('FAILURE')
                        print(result.stdout)
                        print(result.stderr, file=sys.stderr)
                        print(f"command was: {action.command}")
                        stop(daemon_stop_event, db_queue)
                        exit(2)

        except Exception as e:
            print(f"An error occured while running the action for use-case {action.use_case} in phase {action.phase}.{action.subphase}:")
            print(action)
            print(e)
            benchmark_end_time = time.time()
            db_queue.insert('UPDATE benchmark SET successful = ?, end_time = ? WHERE benchmark_sk = ?',
                            ('FALSE', benchmark_end_time, benchmark_sk))
            stop(daemon_stop_event, db_queue)
            exit(1)

    # run the throughput test
    if Phase.SERVING_THROUGHPUT in phases:
        # assemble the streams
        streams = []
        for stream in streams_mapping:
            stream_int = list(map(lambda s: int(s), stream))
            acs = list(filter(lambda a: a.use_case in stream_int, actions_in_order_serving_throuhgput))
            acs = list(sorted(acs, key=lambda a: (a.phase, a.subphase, stream_int.index(a.use_case))))
            streams.append(copy.deepcopy(acs))

        # run the streams
        stream_threads = []
        killall_event = threading.Event()
        i = 0
        index = 0
        for stream in streams:
            name = streams_mapping_keys[i]
            stream_commands = []
            stream_actions = []
            verboses = []
            for action in stream:
                # append stream name to output
                action.command = action.command.replace(STREAM_PLACEHOLDER, name)
                if dry_run:
                    print(action.command)
                else:
                    stream_commands.append(action.command)
                    stream_actions.append(action)
                    verboses.append(verbose)
            if not dry_run:
                s = Stream(index, name, stream_actions, db_queue, benchmark_sk, log_file, verboses, tpcxai_home,
                           killall_event, shell=True)
                stream_threads.append(s)
            i += 1
            index += len(stream_actions)

        if dry_run:
            exit(0)
        else:
            for thread in stream_threads:
                thread.start()

            stream_exceptions = {}
            for thread in stream_threads:
                thread.join()
                if thread.exc:
                    stream_exceptions[thread.name] = thread.exc

            if len(stream_exceptions) > 0:
                print(f"phase STREAMING_THROUGHPUT failed")
                for stream_name, exc in stream_exceptions.items():
                    print(f"stream {stream_name} failed")
                    print(f"{exc}")
                stop(daemon_stop_event, db_queue)
                exit(1)

            for thread, stream in zip(stream_threads, streams):
                for result, action, start_time, end_time, duration in zip(thread.results, stream, thread.start_times, thread.end_times, thread.timings):
                    if result:
                        stream_names.append(thread.name)
                        ucs.append(action.use_case)
                        phases_col.append(action.phase)
                        phases_run_col.append(action.run)
                        sub_phases.append(action.subphase)
                        commands.append(action.command)
                        start_times.append(start_time)
                        end_times.append(end_time)
                        timings.append(duration)
                        std_logs.append(result.stdout)
                        err_logs.append(result.stderr)

    if dry_run:
        exit(0)

    benchmark_end_time = time.time()
    db_queue.insert('UPDATE benchmark SET end_time = ? WHERE benchmark_sk = ?',
                    (benchmark_end_time, benchmark_sk))

    # Extract and print timings
    logs = pd.DataFrame({'stream': stream_names, 'use_case': ucs, 'phase': phases_col, 'phase_run': phases_run_col, 'sub_phase': sub_phases,
                         'command': commands, 'metric': timings, 'std_out': std_logs, 'std_err': err_logs,
                         'start_time': start_times, 'end_time': end_times})
    # change name of phase in presence of multiple runs of said this phase
    logs['phase_name'] = logs['phase'].astype(str) + '_' + logs['phase_run'].astype(str)

    print('========== RESULTS ==========')
    tmp = logs[(logs['sub_phase'] == SubPhase.WORK) &
               ((logs['phase'] == Phase.TRAINING) | (logs['phase'] == Phase.SERVING))]
    has_metric = tmp.size != 0
    if has_metric:
        output = tmp.pivot(index='use_case', columns='phase_name', values='metric')
        for uc in ucs:
            if uc == 0:
                continue
            try:
                quality = qualities[uc]
                qn = quality[0]
                qv = quality[1]
                output.loc[uc, 'qualitity_metric_name'] = qn
                output.loc[uc, 'qualitity_metric_value'] = qv
            except KeyError:
                continue
        print(output)


    # Include datagen as part of TLOAD?
    datagen_time = logs[(logs['sub_phase'] == SubPhase.WORK) & (logs['phase'] == Phase.DATA_GENERATION)]
    datagen_in_tload = workload.get('include_datagen_in_tload',False)==True
    #

    throughput_results = logs[(logs['sub_phase'] == SubPhase.WORK) & (logs['phase'] == Phase.SERVING_THROUGHPUT)]
    throughput_results = throughput_results.groupby(['stream'])['metric'].sum().reset_index()
    if len(throughput_results) > 0:
        print('SERVING THROUGHPUT')
        print(throughput_results)

    i = 1
    while log_file.exists():
        log_file = log_dir / f"tpcxai-sf{scale_factor}-{log_suffix}-{i}.csv"
        i += 1
    logs.to_csv(log_file, index=False)

    if has_metric:
        metrics_file = log_dir / f"tpcxai-metrics-sf{scale_factor}-{log_suffix}.csv"
        i = 1
        while metrics_file.exists():
            metrics_file = log_dir / f"tpcxai-metrics-sf{scale_factor}-{log_suffix}-{i}.csv"
            i += 1
        output.to_csv(metrics_file, index=True)

    aiucpm_metrics_file = log_dir / f"adabench-aiucpm-metrics-sf{scale_factor}-{log_suffix}.csv"
    i=1
    while aiucpm_metrics_file.exists():
        aiucpm_metrics_file = log_dir / f"adabench-aiucpm-metrics-sf{scale_factor}-{log_suffix}-{i}.csv"
        i += 1

    aiucpm_logger = logging.getLogger()
    aiucpm_logger.setLevel(logging.INFO)
    aiucpm_logger.addHandler(logging.FileHandler(aiucpm_metrics_file))
    aiucpm_logger.addHandler(logging.StreamHandler(sys.stdout))
    if not dry_run:
        aiucpm_logger.addHandler(LogDbHandler(benchmark_sk, db_queue))
    AIUCpm_metric = compute_tpcxai_metric(logs,scale_factor,aiucpm_logger,datagen_in_tload,num_streams)

    if AIUCpm_metric > 0:
        aiucpm_logger.info(f'AIUCpm@{scale_factor}={AIUCpm_metric}')
        db_queue.insert('UPDATE benchmark SET successful = ? WHERE benchmark_sk = ?',
                        ('TRUE', benchmark_sk))
    elif not data_gen:
        aiucpm_logger.info(f'Unable to compute AIUCpm@{scale_factor}. One or more required phases or values couldn\'t be executed or computed for this benchmark run.')
        db_queue.insert('UPDATE benchmark SET successful = ? WHERE benchmark_sk = ?',
                        ('FALSE', benchmark_sk))

    stop(daemon_stop_event, db_queue)


def stop(stop_event, db_queue):
    stop_daemons(stop_event)
    db_queue.stop_gracefully()


if __name__ == '__main__':
    main()
