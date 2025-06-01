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

import shutil
import unittest

import numpy as np
from systemds.context import SystemDSContext


class TestListOperations(unittest.TestCase):
    sds: SystemDSContext = None
    temp_dir: str = "tests/list/tmp/readwrite/"

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext()

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()
        shutil.rmtree(cls.temp_dir)

    def test_write_followed_by_read(self):
        """Test write and read of lists variables in python.
        Since we do not support serializing a list (from java to python) yet we
        read and compute each list element when reading again
        """
        m1 = np.array([[1.0, 2.0, 3.0]])
        m1p = self.sds.from_numpy(m1)
        m2 = np.array([[4.0, 5.0, 6.0]])
        m2p = self.sds.from_numpy(m2)
        list_obj = self.sds.array(m1p, m2p)

        path = self.temp_dir + "01"
        list_obj.write(path).compute()
        ret_m1 = self.sds.read(path)[1].as_matrix().compute()
        ret_m2 = self.sds.read(path)[2].as_matrix().compute()
        self.assertTrue(np.allclose(m1, ret_m1))
        self.assertTrue(np.allclose(m2, ret_m2))


if __name__ == "__main__":
    unittest.main(exit=False)
