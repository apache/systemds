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
import copy
import os
import pickle
import tempfile

from systemds.scuro.modality.transformed import TransformedModality


class RepresentationCache:
    """ """

    _instance = None
    _cache_dir = None
    debug = False

    def __new__(cls, debug=False):
        if not cls._instance:
            cls.debug = debug
            cls._instance = super().__new__(cls)
            cls._cache_dir = tempfile.TemporaryDirectory()
            # cls._cache_dir = "representation_cache"
        return cls._instance

    def _generate_cache_filename(self, modality_id, operators):
        """
        Generate a unique filename for an operator based on its name.

        :param operator_name: The name of the operator.
        :return: A full path to the cache file.
        """
        op_names = []
        filename = modality_id
        for operator in operators:
            if isinstance(operator, str):
                op_names.append(operator)
                filename += operator
            else:
                op_names.append(operator.name)
                filename += operator.name

        return os.path.join(self._cache_dir.name, filename), op_names  # _cache_dir.name

    def save_to_cache(self, modality, used_op_names, operators):
        """
        Save data to a cache file.

        :param operator_name: The name of the operator.
        :param data: The data to save.
        """
        filename, op_names = self._generate_cache_filename(
            str(modality.modality_id) + used_op_names, operators
        )
        if not os.path.exists(filename):
            with open(f"{filename}.pkl", "wb") as f:
                pickle.dump(modality.data, f)

            with open(f"{filename}.meta", "wb") as f:
                pickle.dump(modality.metadata, f)

            if self.debug:
                str_names = ", ".join(op_names)
                print(
                    f"Saved data for operator {str(modality.modality_id)}{used_op_names}{str_names} to cache: {filename}"
                )

    def load_from_cache(self, modality, operators):
        """
        Load data from a cache file if it exists.

        :param operator_name: The name of the operator.
        :return: The cached data or None if not found.
        """
        ops = copy.deepcopy(operators)
        filename, op_names = self._generate_cache_filename(
            str(modality.modality_id), ops
        )
        dropped_ops = []
        while not os.path.exists(f"{filename}.pkl"):
            op_names.pop()
            dropped_ops.append(ops.pop())
            if len(ops) < 1:
                break
            filename, op_names = self._generate_cache_filename(
                str(modality.modality_id), ops
            )

        dropped_ops.reverse()
        op_names = "".join(op_names)

        if os.path.exists(f"{filename}.pkl"):
            with open(f"{filename}.meta", "rb") as f:
                metadata = pickle.load(f)

            transformed_modality = TransformedModality(
                modality,
                op_names,
            )
            data = None
            with open(f"{filename}.pkl", "rb") as f:
                if self.debug:
                    print(
                        f"Loaded cached data for operator '{str(modality.modality_id) + op_names}' from {filename}"
                    )
                data = pickle.load(f)
            transformed_modality.data = data
            return transformed_modality, dropped_ops, op_names

        return None, dropped_ops, op_names
