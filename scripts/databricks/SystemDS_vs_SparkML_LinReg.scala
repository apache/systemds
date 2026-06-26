// Databricks notebook source
//-------------------------------------------------------------
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
//
//-------------------------------------------------------------

// SystemDS vs Spark ML: linear regression with categorical encoding
//
// Compares the encode + train time of the same workload:
//   - SystemDS: transformencode (one-hot the categoricals) + lm
//   - Spark ML: StringIndexer + OneHotEncoder + VectorAssembler + LinearRegression
//
// Each engine generates its own dataset of the same shape and cardinality
// (the MLContext DataFrame->Frame bridge is not available on Databricks, so we
// do not share the raw data across engines). Data generation is EXCLUDED from
// the reported time in both engines so the comparison reflects encode + train:
//   - Spark:    the generated DataFrame is cached + materialized first, then
//               only pipeline.fit (encode + train) is timed.
//   - SystemDS: transformencode needs the frame as a live intermediate (an
//               in-memory frame cannot be passed across executes), so we time a
//               full execute (generate + encode + train) and subtract a
//               generation-only execute. encode+train = full - generate.
//
// Each engine is timed on a single run (no warmup), so the numbers reflect what
// a one-shot job actually pays -- including one-time JVM/Catalyst/SystemDS
// compilation -- rather than an idealized steady state.
//
// Prereq: SystemDS.jar installed on the cluster with the SystemDS JVM flags.

// COMMAND ----------

// On a single-node cluster the whole dataset lives in driver memory, so `rows`
// is bounded by the node size (e.g. ~1-1.5M on an i3.xlarge). Scale up the node
// or lower rows if you hit out-of-memory errors.
dbutils.widgets.text("rows", "1000000", "Number of rows")
dbutils.widgets.text("num_numeric", "50", "Numeric feature columns")
dbutils.widgets.text("num_categorical", "4", "Categorical feature columns")
dbutils.widgets.text("cardinality", "20", "Distinct values per categorical")
dbutils.widgets.text("reg", "1e-3", "L2 regularization (lambda)")

val N    = dbutils.widgets.get("rows").toLong
val DNUM = dbutils.widgets.get("num_numeric").toInt
val DCAT = dbutils.widgets.get("num_categorical").toInt
val CARD = dbutils.widgets.get("cardinality").toInt
val REG  = dbutils.widgets.get("reg").toDouble

val numCols = (0 until DNUM).map(i => s"num_$i").toArray
val catCols = (0 until DCAT).map(j => s"cat_$j").toArray
println(s"config: rows=$N numeric=$DNUM categorical=$DCAT cardinality=$CARD reg=$REG")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Spark ML: OneHotEncoder -> LinearRegression (encode + train)

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

// Deterministic linear weights so the target is actually learnable.
val weights = (0 until DNUM).map(i => ((i % 7) - 3) * 0.5)

def genSparkData(rows: Long): DataFrame = {
  var df = spark.range(0, rows).toDF("id")
  for (i <- 0 until DNUM)
    df = df.withColumn(s"num_$i", rand(i.toLong) * 2.0 - 1.0)
  for (j <- 0 until DCAT)
    df = df.withColumn(s"cat_$j", (floor(rand(100L + j) * CARD) + 1).cast("string"))
  val signal = (0 until DNUM).map(i => col(s"num_$i") * lit(weights(i))).reduce(_ + _)
  df.withColumn("y", signal + (rand(999L) * 0.2 - 0.1)).drop("id")
}

def sparkPipeline(): Pipeline = {
  val indexers = catCols.map(c =>
    new StringIndexer().setInputCol(c).setOutputCol(c + "_idx").setHandleInvalid("keep"))
  val ohe = new OneHotEncoder()
    .setInputCols(catCols.map(_ + "_idx")).setOutputCols(catCols.map(_ + "_oh"))
    .setDropLast(false)  // keep all categories, matching SystemDS dummycode
  val assembler = new VectorAssembler()
    .setInputCols(numCols ++ catCols.map(_ + "_oh")).setOutputCol("features")
  val lr = new LinearRegression()
    .setLabelCol("y").setFeaturesCol("features")
    .setRegParam(REG).setElasticNetParam(0.0).setFitIntercept(true)
  new Pipeline().setStages(indexers ++ Array(ohe, assembler, lr))
}

// Time only encode + train: the generated data is cached + materialized first
// (excluded), then pipeline.fit is timed.
def runSpark(rows: Long): (Double, Int, Double) = {
  val df = genSparkData(rows).cache()
  df.count()  // materialize generation outside the timed region
  val t0    = System.nanoTime()
  val model = sparkPipeline().fit(df)
  val secs  = (System.nanoTime() - t0) / 1e9
  val lr    = model.stages.last.asInstanceOf[LinearRegressionModel]
  val res   = (secs, lr.numFeatures, lr.intercept)
  df.unpersist()
  res
}

val (sparkSecs, sparkFeat, sparkIntercept) = runSpark(N)
println(f"Spark ML  encode+train: $sparkSecs%.2f s | features=$sparkFeat intercept=$sparkIntercept%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## SystemDS: transformencode -> lm (encode + train)

// COMMAND ----------

import org.apache.sysds.api.mlcontext._
import org.apache.sysds.api.mlcontext.ScriptFactory._

val ml = new MLContext(sc)
ml.setStatistics(false)
ml.setConfigProperty("sysds.scratch", "/tmp/systemds_scratch")
ml.setConfigProperty("sysds.localtmpdir", "/local_disk0/tmp/systemds")

// transform spec: one-hot (dummycode) the categorical columns. Frame column
// order is [numeric.., categorical.., y]; categoricals are 1-based indices
// DNUM+1 .. DNUM+DCAT. Numeric features and y pass through unchanged.
val catIdx = (DNUM + 1 to DNUM + DCAT).mkString(",")
val spec   = s"""{"ids":true,"dummycode":[$catIdx]}"""

// The data-generation prefix, shared by both scripts below. transformencode
// needs the frame as a live intermediate (an in-memory frame cannot be passed
// across executes), so we cannot pre-build it in a separate execute. Instead we
// time the full execute and a generation-only execute, and subtract. beta is
// random here; only the shape needs to match Spark for a timing comparison.
val genPrefix = """
  Xn    = rand(rows=ROWS, cols=$dn, min=-1.0, max=1.0, seed=7)
  Cids  = floor(rand(rows=ROWS, cols=$dc, min=0.0, max=0.999999, seed=11) * $card) + 1
  beta  = rand(rows=$dn, cols=1, min=-1.0, max=1.0, seed=13)
  noise = rand(rows=ROWS, cols=1, min=-0.1, max=0.1, seed=17)
  y     = Xn %*% beta + noise
"""

// Generation only: materialize the random matrices, nothing else.
val genTemplate = genPrefix + """
  gcheck = sum(Xn) + sum(Cids) + sum(y)
"""

// Full: generation + frame assembly + transformencode + lm.
val fullTemplate = genPrefix + """
  F        = cbind(as.frame(Xn), as.frame(Cids))
  F        = cbind(F, as.frame(y))
  [X0, M]  = transformencode(target=F, spec=spec)
  nc       = ncol(X0)
  yv       = X0[, nc]
  Xv       = X0[, 1:(nc-1)]
  B        = lm(X=Xv, y=yv, icpt=1, reg=reg, verbose=FALSE)
  checksum = sum(B)
  nfeat    = ncol(Xv)
"""

def timeExec(s: Script): (Double, MLResults) = {
  val t0  = System.nanoTime()
  val res = ml.execute(s)
  ((System.nanoTime() - t0) / 1e9, res)
}

def fullScript(rows: Long): Script =
  dml(fullTemplate.replace("ROWS", rows.toString))
    .in("$dn", DNUM).in("$dc", DCAT).in("$card", CARD)
    .in("spec", spec).in("reg", REG).out("checksum", "nfeat")

def genScript(rows: Long): Script =
  dml(genTemplate.replace("ROWS", rows.toString))
    .in("$dn", DNUM).in("$dc", DCAT).in("$card", CARD).out("gcheck")

// encode+train = full - generation (both include identical generation).
def runSysds(rows: Long): (Double, Int, Double) = {
  val (full, res) = timeExec(fullScript(rows))
  val (gen, _)    = timeExec(genScript(rows))
  (math.max(full - gen, 0.0), res.getDouble("nfeat").toInt, res.getDouble("checksum"))
}

val (sysdsSecs, sysdsFeat, sysdsChk) = runSysds(N)
println(f"SystemDS  encode+train: $sysdsSecs%.2f s | features=$sysdsFeat checksum=$sysdsChk%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Result

// COMMAND ----------

val speedup = sparkSecs / sysdsSecs
println(f"""
=== Linear regression (encode + train; data generation excluded) ===
dataset  : rows=$N numeric=$DNUM categorical=$DCAT cardinality=$CARD
Spark ML : $sparkSecs%6.2f s  (features=$sparkFeat)
SystemDS : $sysdsSecs%6.2f s  (features=$sysdsFeat)
Speedup  : $speedup%6.2fx  (Spark / SystemDS)
""")

dbutils.notebook.exit(
  f"rows=$N numeric=$DNUM categorical=$DCAT card=$CARD " +
  f"spark_s=$sparkSecs%.2f sysds_s=$sysdsSecs%.2f speedup=$speedup%.2f")
