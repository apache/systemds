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

from pyspark.context import SparkContext

import numpy as np

from systemml import MLContext, dml, pydml

sc = SparkContext()
ml = MLContext(sc)

class TestAPI(unittest.TestCase):

    def test_output_string(self):
        script = dml("x1 = 'Hello World'").output("x1")
        self.assertEqual(ml.execute(script).get("x1"), "Hello World")

    def test_output_list(self):
        script = """
        x1 = 0.2
        x2 = x1 + 1
        x3 = x1 + 2
        """
        script = dml(script).output("x1", "x2", "x3")
        self.assertEqual(ml.execute(script).get("x1", "x2"), [0.2, 1.2])
        self.assertEqual(ml.execute(script).get("x1", "x3"), [0.2, 2.2])

    def test_output_matrix(self):
        sums = """
        s1 = sum(m1)
        m2 = m1 * 2
        """
        rdd1 = sc.parallelize(["1.0,2.0", "3.0,4.0"])
        script = dml(sums).input(m1=rdd1).output("s1", "m2")
        s1, m2 = ml.execute(script).get("s1", "m2")
        self.assertEqual((s1, repr(m2)), (10.0, "Matrix"))

    def test_matrix_toDF(self):
        sums = """
        s1 = sum(m1)
        m2 = m1 * 2
        """
        rdd1 = sc.parallelize(["1.0,2.0", "3.0,4.0"])
        script = dml(sums).input(m1=rdd1).output("m2")
        m2 = ml.execute(script).get("m2")
        self.assertEqual(repr(m2.toDF()), "DataFrame[__INDEX: double, C1: double, C2: double]")

    def test_matrix_toNumPy(self):
        script = """
        m2 = m1 * 2
        """
        rdd1 = sc.parallelize(["1.0,2.0", "3.0,4.0"])
        script = dml(script).input(m1=rdd1).output("m2")
        m2 = ml.execute(script).get("m2")
        self.assertTrue((m2.toNumPy() == np.array([[2.0, 4.0], [6.0, 8.0]])).all())

    def test_input_single(self):
        script = """
        x2 = x1 + 1
        x3 = x1 + 2
        """
        script = dml(script).input("x1", 5).output("x2", "x3")
        self.assertEqual(ml.execute(script).get("x2", "x3"), [6, 7])

    def test_input(self):
        script = """
        x3 = x1 + x2
        """
        script = dml(script).input(x1=5, x2=3).output("x3")
        self.assertEqual(ml.execute(script).get("x3"), 8)

    def test_rdd(self):
        sums = """
        s1 = sum(m1)
        s2 = sum(m2)
        s3 = 'whatever'
        """
        rdd1 = sc.parallelize(["1.0,2.0", "3.0,4.0"])
        rdd2 = sc.parallelize(["5.0,6.0", "7.0,8.0"])
        script = dml(sums).input(m1=rdd1).input(m2=rdd2).output("s1", "s2", "s3")
        self.assertEqual(ml.execute(script).get("s1", "s2", "s3"), [10.0, 26.0, "whatever"])

    def test_pydml(self):
        script = "A = full('1 2 3 4 5 6 7 8 9', rows=3, cols=3)\nx = toString(A)"
        script = pydml(script).output("x")
        self.assertEqual(
                ml.execute(script).get("x"),
                '1.000 2.000 3.000\n4.000 5.000 6.000\n7.000 8.000 9.000\n'
        )


if __name__ == "__main__":
    unittest.main()
