# ------------------------------------------------------------------------------
#  Copyright 2020 Graz University of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------
import warnings
import unittest
import os
import sys
import re

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")
sys.path.insert(0, path)
from systemds.matrix import Matrix, full, seq
from systemds.utils import helpers

class TestLineageTrace(unittest.TestCase):

    def setUp(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def tearDown(self):
        warnings.filterwarnings(action="ignore",
                                message="unclosed",
                                category=ResourceWarning)

    def test_compare_trace1(self): #test getLineageTrace() on an intermediate
        m = full((5, 10), 4.20)
        m_res = m * 3.1
        m_sum = m_res.sum()       
        with open(os.path.join("tests", "lt.txt"), "r") as file:
            data = file.read()
        file.close()
        self.assertEqual(reVars(m_res.getLineageTrace()), reVars(data))

    def test_compare_trace2(self): #test (lineage=True) as an argument to compute
        m = full((5, 10), 4.20)
        m_res = m * 3.1
        sum, lt = m_res.sum().compute(lineage = True)       
        lt = re.sub(r'\b_mVar\d*\b', '', lt)
        with open(os.path.join("tests", "lt2.txt"), "r") as file:
            data = file.read()
        file.close()
        self.assertEqual(reVars(lt), reVars(data))


def reVars(s: str) -> str:
    s = re.sub(r'\b_mVar\d*\b', '', s)
    s = re.sub(r'\b_Var\d*\b', '', s)
    return s

if __name__ == "__main__":
    unittest.main(exit=False)
    helpers.shutdown()
