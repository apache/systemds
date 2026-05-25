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
# Copyright 2021 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#
import logging
import sqlite3
import time
from pathlib import Path
from typing import Union


class LogDbHandler(logging.Handler):

    def __init__(self, benchmark_id, db_queue):
        logging.Handler.__init__(self)
        self.benchmark_id = benchmark_id
        self.db_queue = db_queue

    def emit(self, record: logging.LogRecord) -> None:
        rec_time = time.time()
        if self.db_queue:
            splits = record.msg.split(":")
            # get metric parts in the form: TLOAD:0.01
            if len(splits) >= 2:
                name, value = splits
                query = 'INSERT INTO performance_metric(benchmark_fk, metric_name, metric_value, metric_time) ' \
                        'VALUES(?, ?, ?, ?)'
                self.db_queue.insert(query, (self.benchmark_id, name, value, rec_time))
            # get final metric in the form: AIUCpm@1.0=10.3
            splits = record.msg.split("=")
            if len(splits) >= 2:
                name, value = splits
                if "@" in name:
                    query = 'INSERT INTO performance_metric(benchmark_fk, metric_name, metric_value, metric_time) ' \
                            'VALUES(?, ?, ?, ?)'
                    self.db_queue.insert(query, (self.benchmark_id, name, value, rec_time))


class LogPerformanceDbHandler():
    """A logging Handler to continuously log performance metrics"""

    # the number of parts in a message that are necessary
    NUM_OF_PARTS = 5

    def __init__(self, benchmark_id, db_queue, level=logging.INFO):
        self.benchmark_id = benchmark_id
        self.db_queue = db_queue

    def emit(self, record: str) -> None:
        parts = list(map(lambda s: s.strip(), record.split(',')))
        if len(parts) >= self.NUM_OF_PARTS:
            q = "INSERT INTO timeseries(benchmark_fk, hostname, name, timestamp, value, unit) VALUES(?, ?, ?, ?, ?, ?)"
            host, timestamp, name, value, unit = parts[:self.NUM_OF_PARTS]
            self.db_queue.dump(q, (self.benchmark_id, host, name, timestamp, value, unit))


class FileAndDBLogger:

    def __init__(self, usercase, log_dir: Union[str, Path], db_queue, max_lines=10000):
        """

        :param log_dir:
        :param db_connection:
        :param max_lines: Maximum number of lines to keep before flushing the logs
        """
        self.log_dir = log_dir
        self.db_queue = db_queue
        self.max_lines = max_lines
        self.std_out = []
        self.std_err = []
        self.usecase = usercase

    def __enter__(self):
        self.file_handle = open(self.log_dir, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.flush()
        self.file_handle.close()

    def log_out(self, msg):
        """
        Collects the log entries until max_bytes threshold is reached
        :param msg: The message or text to write to the log
        :return:
        """
        self.std_out.append(msg)
        self.file_handle.write(f"{msg}")
        self.file_handle.flush()
        if len(self.std_out) >= self.max_lines:
            # flush the log
            self.flush()

    def flush(self):
        text_out = ''.join(self.std_out)

        # write to database
        try:
            last_part = self.db_queue.query('SELECT max(part) FROM log_std_out WHERE use_case_fk = ?',
                                            (self.usecase, )
                                            ).fetchone()[0]
            if last_part:
                current_part = last_part + 1
            else:
                current_part = 1
            self.db_queue.insert('INSERT INTO log_std_out VALUES(?, ?, ?)', (self.usecase, current_part, text_out))
        except sqlite3.Error as e:
            print(e)

        # clear the log
        self.std_out = []

    def last_out(self):
        text = ''.join(self.std_out)
        return text
