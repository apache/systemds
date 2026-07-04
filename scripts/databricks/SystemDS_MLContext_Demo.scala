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

// SystemDS on Databricks: Unity Catalog round-trip
//
// Reads a table from Unity Catalog into SystemDS, runs a DML script over it,
// and writes the result back as a UC table. Prereq: `SystemDS.jar` installed
// on the cluster (DBR 16.4 LTS, Spark 3.5.2 / Scala 2.12) with the SystemDS
// JVM flags.
//
// Everything is configured via the notebook widgets (first cell) so this works
// in any workspace:
//   - catalog / schema / input_table / output_table: where data is read/written
//   - dml_path:   path to a DML script to run (blank = built-in z-score demo)
//   - exec_type:  SystemDS execution mode (DEFAULT = let SystemDS decide)
//
// The DML script contract: it receives the input matrix as `X` and must
// produce a matrix `Y` and a scalar `checksum`.

// COMMAND ----------

// Widgets make the notebook portable: set these per workspace instead of
// hardcoding values. The table location defaults to main.default.
dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("input_table", "systemds_input", "Input table name")
dbutils.widgets.text("output_table", "systemds_output", "Output table name")
// DML script to run. Blank uses the built-in z-score demo below. Otherwise a
// path readable from the cluster driver, e.g. a UC volume, /Workspace, or dbfs:
//   /Volumes/<catalog>/<schema>/<volume>/my_script.dml
dbutils.widgets.text("dml_path", "", "DML script path (blank = built-in demo)")
// SystemDS execution mode. DEFAULT lets SystemDS choose (no forcing).
dbutils.widgets.dropdown("exec_type", "DEFAULT",
  Seq("DEFAULT", "DRIVER", "SPARK", "DRIVER_AND_SPARK"), "Execution type")

val catalog      = dbutils.widgets.get("catalog")
val schema       = dbutils.widgets.get("schema")
val INPUT_TABLE  = s"$catalog.$schema.${dbutils.widgets.get("input_table")}"
val OUTPUT_TABLE = s"$catalog.$schema.${dbutils.widgets.get("output_table")}"
val DML_PATH     = dbutils.widgets.get("dml_path")
val EXEC_TYPE    = dbutils.widgets.get("exec_type")

// COMMAND ----------

import org.apache.sysds.api.mlcontext._
import org.apache.sysds.api.mlcontext.ScriptFactory._
import org.apache.sysds.utils.Statistics
import org.apache.spark.sql.functions._

val ml = new MLContext(sc)
ml.setStatistics(true)
// Only override the execution mode when explicitly requested; DEFAULT leaves
// SystemDS to pick the plan (it will use Spark only when it decides to).
if (EXEC_TYPE != "DEFAULT")
  ml.setExecutionType(MLContext.ExecutionType.valueOf(EXEC_TYPE))
// SystemDS defaults to a relative scratch dir, which the Databricks default
// filesystem rejects ("Path must be absolute"). Pin both to absolute paths.
ml.setConfigProperty("sysds.scratch", "/tmp/systemds_scratch")
ml.setConfigProperty("sysds.localtmpdir", "/local_disk0/tmp/systemds")
println("Spark version: " + sc.version + " | exec_type: " + EXEC_TYPE)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Setup: ensure an input table exists in the catalog

// COMMAND ----------

if (!spark.catalog.tableExists(INPUT_TABLE)) {
  val seed = spark.range(0, 5000).select(
    (rand(1) * 100).as("f1"),
    (rand(2) * 10  + 5).as("f2"),
    (rand(3) - 0.5).as("f3"),
    (rand(4) * 1000).as("f4"))
  seed.write.mode("overwrite").saveAsTable(INPUT_TABLE)
}
println(s"input table $INPUT_TABLE rows = ${spark.table(INPUT_TABLE).count()}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 1. Read a table from the catalog into SystemDS

// COMMAND ----------

val inDF = spark.table(INPUT_TABLE)
  .select(col("f1").cast("double"), col("f2").cast("double"),
          col("f3").cast("double"), col("f4").cast("double"))

// Built-in fallback: standardize the columns (z-score). Used when no dml_path
// widget is set. A custom script must read `X` and produce `Y` and `checksum`.
val defaultScript = """
  n        = nrow(X)
  mu       = colMeans(X)
  Xc       = X - mu
  variance = colSums(Xc^2) / (n - 1)
  sigma    = sqrt(variance)
  Y        = Xc / sigma
  checksum = sum(Y)
  print("standardized " + nrow(X) + " x " + ncol(X) + " matrix")
"""

val baseScript = if (DML_PATH.nonEmpty) {
  println(s"running DML from $DML_PATH")
  dmlFromFile(DML_PATH)
} else {
  println("running built-in z-score demo script")
  dml(defaultScript)
}
val script = baseScript.in("X", inDF).out("Y", "checksum")

val res      = ml.execute(script)
val checksum = res.getDouble("checksum")
val outDF    = res.getDataFrameDoubleNoIDColumn("Y")
println(s"checksum(Y) = $checksum  (≈0 for standardized data)")

// COMMAND ----------

// MAGIC %md
// MAGIC ## 2. Write the SystemDS result back to the catalog

// COMMAND ----------

outDF.write.mode("overwrite").saveAsTable(OUTPUT_TABLE)
val outRows = spark.table(OUTPUT_TABLE).count()
val outCols = spark.table(OUTPUT_TABLE).columns.length
println(s"wrote $OUTPUT_TABLE: $outRows rows x $outCols cols")

// COMMAND ----------

val spExecuted = Statistics.getNoOfExecutedSPInst()
dbutils.notebook.exit(
  s"spark=${sc.version} in=$INPUT_TABLE out=$OUTPUT_TABLE " +
  s"out_rows=$outRows out_cols=$outCols checksum=$checksum sp_executed=$spExecuted")
