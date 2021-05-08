// Databricks notebook source
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */


// COMMAND ----------

// MAGIC %md # Apache SystemDS on Databricks in 5 minutes

// COMMAND ----------

// MAGIC %md ## Create a quickstart cluster
// MAGIC 
// MAGIC 1. In the sidebar, right-click the **Clusters** button and open the link in a new window.
// MAGIC 1. On the Clusters page, click **Create Cluster**.
// MAGIC 1. Name the cluster **Quickstart**.
// MAGIC 1. In the Databricks Runtime Version drop-down, select **6.4 (Scala 2.11, Spark 2.4.5)**.
// MAGIC 1. Click **Create Cluster**.
// MAGIC 1. Attach `SystemDS.jar` file to the libraries

// COMMAND ----------

// MAGIC %md ## Attach the notebook to the cluster and run all commands in the notebook
// MAGIC 
// MAGIC 1. Return to this notebook. 
// MAGIC 1. In the notebook menu bar, select **<img src="http://docs.databricks.com/_static/images/notebooks/detached.png"/></a> > Quickstart**.
// MAGIC 1. When the cluster changes from <img src="http://docs.databricks.com/_static/images/clusters/cluster-starting.png"/></a> to <img src="http://docs.databricks.com/_static/images/clusters/cluster-running.png"/></a>, click **<img src="http://docs.databricks.com/_static/images/notebooks/run-all.png"/></a> Run All**.

// COMMAND ----------

// MAGIC %md ## Load SystemDS MLContext API

// COMMAND ----------

import org.apache.sysds.api.mlcontext._
import org.apache.sysds.api.mlcontext.ScriptFactory._
val ml = new MLContext(spark)

// COMMAND ----------

val habermanUrl = "http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
val habermanList = scala.io.Source.fromURL(habermanUrl).mkString.split("\n")
val habermanRDD = sc.parallelize(habermanList)
val habermanMetadata = new MatrixMetadata(306, 4)
val typesRDD = sc.parallelize(Array("1.0,1.0,1.0,2.0"))
val typesMetadata = new MatrixMetadata(1, 4)
val scriptUrl = "https://raw.githubusercontent.com/apache/systemds/master/scripts/algorithms/Univar-Stats.dml"
val uni = dmlFromUrl(scriptUrl).in("A", habermanRDD, habermanMetadata).in("K", typesRDD, typesMetadata).in("$CONSOLE_OUTPUT", true)
ml.execute(uni)

// COMMAND ----------

// MAGIC %md #### Create a neural network layer with (R-like) DML language

// COMMAND ----------

val s = """
  source("scripts/nn/layers/relu.dml") as relu;
  X = rand(rows=100, cols=10, min=-1, max=1);
  R1 = relu::forward(X);
  R2 = max(X, 0);
  R = sum(R1==R2);
  """

val ret = ml.execute(dml(s).out("R")).getScalarObject("R").getDoubleValue();
