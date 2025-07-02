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
import shutil
import unittest

from systemds.context import SystemDSContext

os.environ["SYSDS_QUIET"] = "1"

test_dir = os.path.join("tests", "lineage")
temp_dir = os.path.join(test_dir, "temp")
trace_test_1 = os.path.join(test_dir, "trace1.dml")


class TestLineageTrace(unittest.TestCase):
    sds: SystemDSContext = None

    @classmethod
    def setUpClass(cls):
        cls.sds = SystemDSContext(capture_stdout=True)

    @classmethod
    def tearDownClass(cls):
        cls.sds.close()

    def tearDown(self):
        shutil.rmtree(temp_dir, ignore_errors=True)

    @unittest.skipIf(
        "SYSTEMDS_ROOT" not in os.environ,
        "The test is skipped if SYSTEMDS_ROOT is not set, this is required for this tests since it use the bin/systemds file to execute a reference",
    )
    def test_compare_trace1(self):  # test getLineageTrace() on an intermediate
        m = self.sds.full((10, 10), 1)
        m_res = m + m

        python_trace = [
            x.strip().split("°") for x in m_res.get_lineage_trace().split("\n")
        ]

        sysds_trace = self.create_execute_and_trace_dml(trace_test_1)

        # It is not guarantied, that the two lists 100% align to be the same.
        # Therefore for now, we only compare if the command is the same, in same order.
        python_trace_commands = [x[:1] for x in python_trace]
        dml_script_commands = [x[:1] for x in sysds_trace]
        if len(python_trace_commands) == 0:
            self.fail("Error in pythonscript execution")
        if len(dml_script_commands) == 0:
            self.fail("Error in DML script execution")

        self.assertEqual(python_trace_commands[0], dml_script_commands[0])

    def create_execute_and_trace_dml(self, script: str):
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Call SYSDS!
        result_file_name = temp_dir + "/tmp_res.txt"
        command = "systemds " + script + " > " + result_file_name + " 2> /dev/null"
        status = os.system(command)
        if status < 0:
            self.fail("systemds call failed.")
        return parse_trace(result_file_name)


def parse_trace(path: str):
    data = []
    with open(path, "r") as log:
        for line in log:
            if "°" in line:
                data.append(line.strip().split("°"))

    # Remove the last 4 lines of the System output because they are after lintrace.
    return data


if __name__ == "__main__":
    unittest.main(exit=False)
