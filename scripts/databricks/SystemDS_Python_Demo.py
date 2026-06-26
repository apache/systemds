# Databricks notebook source
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

# SystemDS Python API on Databricks
#
# The SystemDS Python API (`systemds`) spawns its own JVM (via py4j) and runs
# SystemDS in-process on the driver. This is independent of the cluster's
# SparkContext, so it does not need the cluster JAR library — just the pip
# package (which bundles its own SystemDS jar).

# COMMAND ----------

# MAGIC %pip install systemds

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
from systemds.context import SystemDSContext

with SystemDSContext() as sds:
    # 1) generate + matmul + aggregate entirely inside the SystemDS JVM
    X = sds.rand(rows=2000, cols=500, min=0.0, max=1.0, seed=42)
    gram_sum = (X @ X.t()).sum().compute()

    # 2) round-trip a numpy array through SystemDS
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    M = sds.from_numpy(a)
    col_sums = M.sum(axis=0).compute()

print("gram_sum =", gram_sum)
print("col_sums =", col_sums.tolist() if hasattr(col_sums, "tolist") else col_sums)

dbutils.notebook.exit(
    f"python_api_ok gram_sum={gram_sum} col_sums={np.ravel(col_sums).tolist()}")
