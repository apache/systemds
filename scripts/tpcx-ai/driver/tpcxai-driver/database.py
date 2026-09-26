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


import queue
import sqlite3
from threading import Thread, BoundedSemaphore


class DatabaseQueue:
    """Single point of contact with the SQLite database. A background thread is started that will handle access to the
    database to prevent concurrency issues. The queries are queued and handled in a FIFO fashion. Internally there are
    two queue that handle different types of database requests.
    """

    def __init__(self, database_path):
        self.database_path = database_path
        self.queue = queue.Queue()
        self.insert_queue = queue.Queue()
        self.stop_signaled = False
        self.db_semaphore = BoundedSemaphore(2)
        self.db_thread = Thread(target=self._db_worker, args=(self.queue, ), daemon=True)
        self.db_thread.start()
        self.db_thread_insert = Thread(target=self._db_worker, args=(self.insert_queue, ))
        self.db_thread_insert.start()
        db_uri = database_path.as_uri() + '?mode=ro'
        self.query_connection = sqlite3.connect(db_uri, uri=True, check_same_thread=False)

    def query(self, query, *args) -> sqlite3.Cursor:
        """
        Read only access to the underlying database, blocks and returns the result cursor.
        :param query: The SELECT query to be executed
        :param args: The parameters for the query
        :return: Cursor with the result of the given query
        """
        try:
            self.insert_queue.join()
            res = self.query_connection.execute(query, *args)
            self.query_connection.commit()
            return res
        except Exception as e:
            raise RuntimeError(f"Error running {query} with parameters: {args}") from e

    def insert(self, query, params=[]):
        """
        Executes a DB write. All writes are queued and executed by a single background thread to prevent concurrency
        issues. Statements given are guaranteed to be executed eventually. That is, the main thread waits until all
        request are fulfilled.
        :param query:
        :param params:
        """
        query_item = (query, params)
        self.insert_queue.put(query_item)

    def dump(self, query, params=None):
        """
        Executes a DB write (non-guaranteed). All writes are queued and executed by an exclusive background thread.
        Statements given are *not* guaranteed to be executed. That is the background thread is killed when the main
        thead is killed and all remaining statements are left unfinished.
        :param query:
        :param params:
        """
        query_item = (query, params)
        self.queue.put(query_item)

    def stop_gracefully(self):
        """
        Wait for the insert queue to finish.
        :return:
        """
        self.insert_queue.join()
        self.stop_signaled = True

    def stop(self):
        """
        Kill all background threads immediately.
        :return:
        """
        self.stop_signaled = True

    def wait_until_finished(self):
        self.insert_queue.join()

    def _db_worker(self, sql_queue: queue.Queue):
        connection = sqlite3.connect(str(self.database_path))
        while not self.stop_signaled:
            try:
                query, params = sql_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                # guard the db with a semaphore to only allow one write at a time
                with self.db_semaphore:
                    connection.execute(query, params)
                    connection.commit()
            except sqlite3.Error as e:
                raise RuntimeError(f"An error occurred when running {query} with params: {params}") from e
            finally:
                sql_queue.task_done()

        connection.close()
