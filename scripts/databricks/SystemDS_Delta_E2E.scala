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

// End-to-end Delta -> linear regression, on the SAME Delta table for both engines:
//   - SystemDS: read(format="delta") as a frame -> transformencode -> lm
//   - Spark ML: spark.read delta -> StringIndexer + OneHotEncoder + LinearRegression
//
// SystemDS reads the Delta table natively through the Spark-free Delta Kernel
// (FrameReaderDelta), so the whole pipeline runs inside the SystemDS runtime.
// The timed region is end-to-end: read + encode + train, for both engines.
//
// Prereqs (handled by deploy.sh):
//   - SystemDS.jar installed on the cluster, with the SystemDS JVM flags.
//   - io.delta:delta-kernel-api + delta-kernel-defaults installed as Maven
//     libraries (Delta Kernel is NOT on the DBR classpath and is NOT bundled in
//     SystemDS.jar).

// COMMAND ----------

// On a single-node cluster SystemDS reads the whole table into driver memory, so
// both `rows` and the encoded width (num_numeric + num_categorical*cardinality)
// are bounded by the node size. On an i3.xlarge (~30 GB) ~10M rows OOMs in
// transformencode; scale up the node or lower these on OOM.
//
// What drives the SystemDS-vs-Spark gap is encoding complexity, not row count:
// more categoricals / higher cardinality blow up Spark's StringIndexer +
// OneHotEncoder (each a shuffle stage), while SystemDS dummycodes in-memory.
// Raw rows instead favor Spark (it parallelizes across cores; single-node CP
// does not). Defaults below are deliberately encode-heavy (700 features) so the
// difference is visible; the baseline 50/4/20 config is roughly a tie at 1M rows.
dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("volume", "systemds", "Volume (table is written under it)")
dbutils.widgets.text("rows", "1000000", "Number of rows")
dbutils.widgets.text("num_numeric", "100", "Numeric feature columns")
dbutils.widgets.text("num_categorical", "20", "Categorical feature columns")
dbutils.widgets.text("cardinality", "30", "Distinct values per categorical")
dbutils.widgets.text("reg", "1e-3", "L2 regularization (lambda)")
dbutils.widgets.dropdown("recreate", "true", Seq("true", "false"), "Recreate the Delta table")
dbutils.widgets.dropdown("statistics", "true", Seq("true", "false"), "Print SystemDS statistics")

val CATALOG = dbutils.widgets.get("catalog")
val SCHEMA  = dbutils.widgets.get("schema")
val VOLUME  = dbutils.widgets.get("volume")
val N       = dbutils.widgets.get("rows").toLong
val DNUM    = dbutils.widgets.get("num_numeric").toInt
val DCAT    = dbutils.widgets.get("num_categorical").toInt
val CARD    = dbutils.widgets.get("cardinality").toInt
val REG     = dbutils.widgets.get("reg").toDouble
val RECREATE = dbutils.widgets.get("recreate").toBoolean
val STATS    = dbutils.widgets.get("statistics").toBoolean

val numCols = (0 until DNUM).map(i => s"num_$i").toArray
val catCols = (0 until DCAT).map(j => s"cat_$j").toArray

// The table lives on a UC volume (FUSE-mounted locally at /Volumes/...). Spark
// reads it via the same path; SystemDS reads the local FUSE mount with an
// explicit file: scheme so the Delta Kernel's Hadoop engine uses the local
// filesystem rather than the cluster default (dbfs).
val tablePath = s"/Volumes/$CATALOG/$SCHEMA/$VOLUME/delta_e2e"
val sysdsPath = "file:" + tablePath
println(s"config: rows=$N numeric=$DNUM categorical=$DCAT cardinality=$CARD reg=$REG")
println(s"table : $tablePath")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Setup: materialize the dataset as a Delta table (once)

// COMMAND ----------

import org.apache.spark.sql.functions._

// Deterministic linear weights so the target is actually learnable.
val weights = (0 until DNUM).map(i => ((i % 7) - 3) * 0.5)

def writeDeltaTable(): Unit = {
  var df = spark.range(0, N).toDF("id")
  for (i <- 0 until DNUM)
    df = df.withColumn(s"num_$i", rand(i.toLong) * 2.0 - 1.0)
  for (j <- 0 until DCAT)
    df = df.withColumn(s"cat_$j", (floor(rand(100L + j) * CARD) + 1).cast("string"))
  val signal = (0 until DNUM).map(i => col(s"num_$i") * lit(weights(i))).reduce(_ + _)
  // Column order written to Delta is [numeric.., categorical.., y]; SystemDS
  // relies on this order for the transform spec and target column.
  val out = df.withColumn("y", signal + (rand(999L) * 0.2 - 0.1)).drop("id")
    .select((numCols ++ catCols ++ Array("y")).map(col): _*)
  // overwriteSchema so re-running with a different feature count replaces an
  // existing table whose schema no longer matches.
  out.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(tablePath)
}

val exists = try { dbutils.fs.ls(tablePath); true } catch { case _: Throwable => false }
if (RECREATE || !exists) {
  println(">> writing Delta table ...")
  writeDeltaTable()
}
val tblRows = spark.read.format("delta").load(tablePath).count()
println(s">> Delta table ready: $tblRows rows at $tablePath")

// COMMAND ----------

// MAGIC %md
// MAGIC ## SystemDS: read Delta (native Kernel) -> transformencode -> lm

// COMMAND ----------

import org.apache.sysds.api.mlcontext._
import org.apache.sysds.api.mlcontext.ScriptFactory._

val ml = new MLContext(sc)
// With statistics on, SystemDS prints a per-instruction breakdown (heavy hitters)
// after execute: the Delta read shows up as cache acquire-read (ACQr) time, plus
// transformencode and the lm operators (m_lm/tsmm/solve). Useful to see where the
// end-to-end time actually goes.
ml.setStatistics(STATS)
ml.setStatisticsMaxHeavyHitters(25)
ml.setConfigProperty("sysds.scratch", "/tmp/systemds_scratch")
ml.setConfigProperty("sysds.localtmpdir", "/local_disk0/tmp/systemds")
// Force single-node (CP) execution. The native Delta frame reader
// (FrameReaderDelta) is a control-program reader; under Spark execution the
// frame read is distributed and bypasses it (failing to parse the Delta parquet
// files). On a single-node cluster CP execution is the intended mode anyway.
ml.setExecutionType(MLContext.ExecutionType.DRIVER)

// transform spec: one-hot (dummycode) the categorical columns. Delta column
// order is [numeric.., categorical.., y]; categoricals are 1-based indices
// DNUM+1 .. DNUM+DCAT. Numeric features and y pass through unchanged.
val catIdx = (DNUM + 1 to DNUM + DCAT).mkString(",")
val spec   = s"""{"ids":true,"dummycode":[$catIdx]}"""

// Whole pipeline in one script: native Delta frame read -> encode -> train.
val e2eDml = """
  F        = read($path, data_type="frame", format="delta")
  [X, M]   = transformencode(target=F, spec=spec)
  nc       = ncol(X)
  yv       = X[, nc]
  Xv       = X[, 1:(nc-1)]
  B        = lm(X=Xv, y=yv, icpt=1, reg=reg, verbose=FALSE)
  checksum = sum(B)
  nfeat    = ncol(Xv)
  nrows    = nrow(X)
"""

def runSysds(path: String): (Double, Int, Long, Double) = {
  val script = dml(e2eDml)
    .in("$path", path).in("spec", spec).in("reg", REG)
    .out("checksum", "nfeat", "nrows")
  val t0  = System.nanoTime()
  val res = ml.execute(script)
  val secs = (System.nanoTime() - t0) / 1e9
  (secs, res.getDouble("nfeat").toInt, res.getDouble("nrows").toLong, res.getDouble("checksum"))
}

val (sysdsSecs, sysdsFeat, sysdsRows, sysdsChk) = runSysds(sysdsPath)
println(f"SystemDS  read+encode+train: $sysdsSecs%.2f s | rows=$sysdsRows features=$sysdsFeat checksum=$sysdsChk%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Spark ML: read Delta -> OneHotEncoder -> LinearRegression

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}

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

// End-to-end: time the lazy Delta read + encode + train together (the read is
// triggered inside pipeline.fit), symmetric to the SystemDS execute above.
def runSpark(path: String): (Double, Int, Double) = {
  val t0    = System.nanoTime()
  val df    = spark.read.format("delta").load(path)
  val model = sparkPipeline().fit(df)
  val secs  = (System.nanoTime() - t0) / 1e9
  val lr    = model.stages.last.asInstanceOf[LinearRegressionModel]
  (secs, lr.numFeatures, lr.intercept)
}

val (sparkSecs, sparkFeat, sparkIntercept) = runSpark(tablePath)
println(f"Spark ML  read+encode+train: $sparkSecs%.2f s | features=$sparkFeat intercept=$sparkIntercept%.4f")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Result

// COMMAND ----------

val speedup = sparkSecs / sysdsSecs
println(f"""
=== E2E Delta -> linear regression (read + encode + train) ===
dataset  : rows=$N numeric=$DNUM categorical=$DCAT cardinality=$CARD
table    : $tablePath
Spark ML : $sparkSecs%6.2f s  (features=$sparkFeat)
SystemDS : $sysdsSecs%6.2f s  (features=$sysdsFeat, rows read=$sysdsRows)
Speedup  : $speedup%6.2fx  (Spark / SystemDS)
""")

dbutils.notebook.exit(
  f"rows=$N numeric=$DNUM categorical=$DCAT card=$CARD " +
  f"spark_s=$sparkSecs%.2f sysds_s=$sysdsSecs%.2f speedup=$speedup%.2f " +
  f"sysds_features=$sysdsFeat sysds_rows=$sysdsRows")
