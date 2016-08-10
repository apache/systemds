#!/usr/bin/python
#-------------------------------------------------------------
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
#-------------------------------------------------------------
import unittest

from pyspark.sql import SQLContext
from pyspark.context import SparkContext

from SystemML import Dml, MLContext

sc = SparkContext()
ml = MLContext(sc)

class TestAPI(unittest.TestCase):

    def test_output_string(self):
        dml = Dml("x1 = 'Hello World'").out("x1")
        ml.execute(dml)
        self.assertEqual(ml.execute(dml).get("x1"), "Hello World")

    def test_output_list(self):
        script = """
        x1 = 0.2
        x2 = x1 + 1
        x3 = x1 + 2
        """
        dml = Dml(script).out("x1", "x2", "x3")
        self.assertEqual(ml.execute(dml).get("x1", "x2"), [0.2, 1.2])
        self.assertEqual(ml.execute(dml).get("x1", "x3"), [0.2, 2.2])

    def test_input_single(self):
        script = """
        x2 = x1 + 1
        x3 = x1 + 2
        """
        dml = Dml(script).input("x1", 5).out("x2", "x3")
        self.assertEqual(ml.execute(dml).get("x2", "x3"), [6, 7])

    def test_input(self):
        script = """
        x3 = x1 + x2
        """
        dml = Dml(script).input(x1=5, x2=3).out("x3")
        self.assertEqual(ml.execute(dml).get("x3"), 8)

    def test_rdd(self):
        sums = """
        s1 = sum(m1)
        s2 = sum(m2)
        s3 = 'whatever'
        """
        rdd1 = sc.parallelize(["1.0,2.0", "3.0,4.0"])
        rdd2 = sc.parallelize(["5.0,6.0", "7.0,8.0"])
        dml = Dml(sums).input(m1=rdd1).input(m2=rdd2).out("s1", "s2", "s3")
        self.assertEqual(
            ml.execute(dml).get("s1", "s2", "s3"), [10.0, 26.0, "whatever"])


if __name__ == "__main__":
    unittest.main()
