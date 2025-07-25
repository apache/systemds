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
import logging
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

from systemds.context import SystemDSContext


class TestContextCreation(unittest.TestCase):

    def test_random_port_debug(self):
        SystemDSContext._logging_initialized = False

        stderr_buffer = io.StringIO()

        with redirect_stderr(stderr_buffer):
            sds1 = SystemDSContext(logging_level=10)
            sds1.close()

        err = stderr_buffer.getvalue()
        print("Captured STDERR:\n", err)
        print("END OF STDERR\n")

        self.assertIn("DEBUG SystemDSContext: Logging setup done", err)

    def test_random_port_debug2(self):
        SystemDSContext._logging_initialized = False

        stderr_buffer = io.StringIO()

        with redirect_stderr(stderr_buffer):
            sds1 = SystemDSContext()
            sds1.close()

            err = stderr_buffer.getvalue()
            print("\nCaptured STDERR (ctx1):\n", err)
            print("END OF STDERR\n")

            # clear the buffer
            stderr_buffer.seek(0)
            stderr_buffer.truncate(0)

            sds2 = SystemDSContext(logging_level=10)
            sds2.close()

        err = stderr_buffer.getvalue()
        print("\nCaptured STDERR (ctx2):\n", err)
        print("END OF STDERR\n")

        self.assertIn("DEBUG SystemDSContext: Logging setup done", err)

    def test_random_port_debug3(self):
        SystemDSContext._logging_initialized = False

        sds1 = SystemDSContext()
        sds1.close()
        stderr_buffer = io.StringIO()

        with redirect_stderr(stderr_buffer):
            sds2 = SystemDSContext(logging_level=10)
            sds2.close()

        err = stderr_buffer.getvalue()
        print("\nCaptured STDERR (ctx2):\n", err)
        print("END OF STDERR\n")

        self.assertIn("DEBUG SystemDSContext: Logging setup done", err)

    def test_random_port(self):
        sds1 = SystemDSContext()
        sds1.close()

    def test_two_random_port(self):
        sds1 = SystemDSContext(logging_level=20)
        sds2 = SystemDSContext(logging_level=20)
        sds1.close()
        sds2.close()

    def test_same_port(self):
        # Same port should graciously change port
        sds1 = SystemDSContext(port=9415)
        sds2 = SystemDSContext(port=9415)
        sds1.close()
        sds2.close()

    def test_create_10_contexts(self):
        # Creating multiple contexts and closing them should be no problem.
        for _ in range(0, 10):
            SystemDSContext().close()

    def test_create_multiple_context(self):
        # Creating multiple contexts in sequence but open at the same time is okay.
        a = SystemDSContext()
        b = SystemDSContext()
        c = SystemDSContext()
        d = SystemDSContext()

        a.close()
        b.close()
        c.close()
        d.close()


if __name__ == "__main__":
    unittest.main(exit=False)
