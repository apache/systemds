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

import unittest
from types import SimpleNamespace

from systemds.scuro.drsearch.algebra import AlgebraConfig, order_invariant
from systemds.scuro.models.model import Model


def _tasks(*flags):
    return [
        SimpleNamespace(model=SimpleNamespace(feature_order_invariant=f)) for f in flags
    ]


class TestOrderInvariant(unittest.TestCase):
    def test_model_default_is_unknown(self):
        self.assertIsNone(Model("m").feature_order_invariant)

    def test_cases(self):
        cases = [
            (None, _tasks(True, True), True),
            (None, _tasks(True, None), False),
            (None, _tasks(True, False), False),
            (None, [], False),
            (None, [SimpleNamespace(model=object())], False),
            (True, _tasks(False), True),
            (False, _tasks(True), False),
        ]
        for override, tasks, expected in cases:
            with self.subTest(override=override, tasks=tasks):
                config = AlgebraConfig(feature_order_invariant=override)
                self.assertIs(order_invariant(config, tasks), expected)


if __name__ == "__main__":
    unittest.main()
