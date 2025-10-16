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
import threading


class Identifier:
    """ """

    _instance = None
    id = -1

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def new_id(self):  # TODO: make threadsafe when parallelizing
        self.id += 1
        return self.id


class IdGenerator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._ctr = 0
                    cls._instance._ctr_lock = threading.Lock()
        return cls._instance

    def next(self) -> int:
        with self._instance._ctr_lock:
            self._instance._ctr += 1
            n = self._instance._ctr
        return n


get_op_id = IdGenerator().next
get_node_id = IdGenerator().next
