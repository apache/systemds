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

package org.apache.sysml.api.test.ml

import org.apache.sysml.api.ml.regression.LinearRegressionDS
import java.io.File.separator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext

object LinearRegressionDSExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val sparkConf = new SparkConf()
      .setAppName("LInear SVM Exanple ")
      .setMaster("local");

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
    val fPath = this.getClass()
                    .getResource("/sample_libsvm_data.txt")
                    .getPath()
    val data = sqlContext
      .createDataFrame(MLUtils.loadLibSVMFile(sc, fPath).collect)
      .toDF("label", "features")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed = 12345)
    
    val lr = new LinearRegressionDS(sc)
    
    val pipeline = new Pipeline().setStages(Array(lr))
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .build()
    
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(10)
    val model = crossval.fit(training)
    model.transform(test)
      .select("label", "features", "prediction")
      .show
  }
}