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

// MAGIC %md # Apache SystemDS on Databricks

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

// MAGIC %md ### Create a neural network layer with (R-like) DML language

// COMMAND ----------

val s = """
  source("scripts/nn/layers/relu.dml") as relu;
  X = rand(rows=100, cols=10, min=-1, max=1);
  R1 = relu::forward(X);
  R2 = max(X, 0);
  R = sum(R1==R2);
  """

val ret = ml.execute(dml(s).out("R")).getScalarObject("R").getDoubleValue();

// COMMAND ----------

// MAGIC %md ### Recommendation with Amazon review dataset

// COMMAND ----------

import java.net.URL
import java.io.File
import org.apache.commons.io.FileUtils

FileUtils.copyURLToFile(new URL("http://snap.stanford.edu/data/amazon0601.txt.gz"), new File("/tmp/amazon0601.txt.gz"))

// COMMAND ----------

// MAGIC %sh
// MAGIC gunzip -d /tmp/amazon0601.txt.gz

// COMMAND ----------

// To list the file system files. For more https://docs.databricks.com/data/filestore.html
// File system: display(dbutils.fs.ls("file:/tmp"))
// DBFS: display(dbutils.fs.ls("."))

dbutils.fs.mv("file:/tmp/amazon0601.txt", "dbfs:/tmp/amazon0601.txt")

// COMMAND ----------

display(dbutils.fs.ls("/tmp"))
// display(dbutils.fs.ls("file:/tmp"))

// COMMAND ----------

// move temporary files to databricks file system (DBFS)
// dbutils.fs.mv("file:/databricks/driver/amazon0601.txt", "dbfs:/tmp/amazon0601.txt") 
val df = spark.read.format("text").option("inferSchema", "true").option("header","true").load("dbfs:/tmp/amazon0601.txt")
display(df)

// COMMAND ----------

// MAGIC %py
// MAGIC 
// MAGIC # The scala data processing pipeline can also be
// MAGIC # implemented in python as shown in this block
// MAGIC 
// MAGIC # 
// MAGIC # import pyspark.sql.functions as F
// MAGIC # # https://spark.apache.org/docs/latest/sql-ref.html
// MAGIC 
// MAGIC # dataPath = "dbfs:/tmp/amazon0601.txt"
// MAGIC 
// MAGIC # X_train = (sc.textFile(dataPath)
// MAGIC #     .filter(lambda l: not l.startswith("#"))
// MAGIC #     .map(lambda l: l.split("\t"))
// MAGIC #     .map(lambda prods: (int(prods[0]), int(prods[1]), 1.0))
// MAGIC #     .toDF(("prod_i", "prod_j", "x_ij"))
// MAGIC #     .filter("prod_i < 500 AND prod_j < 500") # Filter for memory constraints
// MAGIC #     .cache())
// MAGIC 
// MAGIC # max_prod_i = X_train.select(F.max("prod_i")).first()[0]
// MAGIC # max_prod_j = X_train.select(F.max("prod_j")).first()[0]
// MAGIC # numProducts = max(max_prod_i, max_prod_j) + 1 # 0-based indexing
// MAGIC # print("Total number of products: {}".format(numProducts))

// COMMAND ----------

// Reference: https://spark.apache.org/docs/latest/rdd-programming-guide.html
val X_train = (sc.textFile("dbfs:/tmp/amazon0601.txt").filter(l => !(l.startsWith("#"))).map(l => l.split("\t"))
                  .map(prods => (prods(0).toLong, prods(1).toLong, 1.0))
                  .toDF("prod_i", "prod_j", "x_ij")
                  .filter("prod_i < 500 AND prod_j < 500") // filter for memory constraints
                  .cache())

display(X_train)

// COMMAND ----------

// MAGIC %md #### Poisson Nonnegative Matrix Factorization

// COMMAND ----------

# Poisson Nonnegative Matrix Factorization

val pnmf = """
# data & args
X = X+1 # change product IDs to be 1-based, rather than 0-based
V = table(X[,1], X[,2])
size = ifdef($size, -1)
if(size > -1) {
    V = V[1:size,1:size]
}

n = nrow(V)
m = ncol(V)
range = 0.01
W = Rand(rows=n, cols=rank, min=0, max=range, pdf="uniform")
H = Rand(rows=rank, cols=m, min=0, max=range, pdf="uniform")
losses = matrix(0, rows=max_iter, cols=1)

# run PNMF
i=1
while(i <= max_iter) {
  # update params
  H = (H * (t(W) %*% (V/(W%*%H))))/t(colSums(W)) 
  W = (W * ((V/(W%*%H)) %*% t(H)))/t(rowSums(H))
  
  # compute loss
  losses[i,] = -1 * (sum(V*log(W%*%H)) - as.scalar(colSums(W)%*%rowSums(H)))
  i = i + 1;
}
  """

val ret = ml.execute(dml(pnmf).in("X", X_train).in("max_iter", 100).in("rank", 10).out("W").out("H").out("losses"));

// COMMAND ----------

val W = ret.getMatrix("W")
val H = ret.getMatrix("H")
val losses = ret.getMatrix("losses")

// COMMAND ----------

val lossesDF = losses.toDF().sort("__INDEX")
display(lossesDF)
