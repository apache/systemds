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

import org.apache.sysml.api.ml.regression.CoxRegression
import org.apache.sysml.api.ml.evaluation.SurvivalAnalysisEvaluator
import java.io.File.separator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SQLContext}


object CoxExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val sparkConf = new SparkConf()
                .setAppName("Cox Exanple ")
                .setMaster("local");
    
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////                      Read Data                      ////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    val x = sc.textFile("src" + separator + "test" + separator + "resources" + separator + "Cox" + separator + "X")
    val y = sc.textFile("src" + separator + "test" + separator + "resources" + separator + "Cox" + separator + "Y")

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////                   Data Preparation                  ////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    val training: DataFrame = x.zipWithIndex().map { 
      case (line, i) => (i + 1, Vectors.dense(line.split(",").map(_.toDouble))) 
    }.toDF("id", "dataset")
    
    val test: DataFrame = y.zipWithIndex().map { 
      case (line, i) => (i + 1, Vectors.dense(line.split(",").map(_.toDouble))) 
    }.toDF("id", "dataset")
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////            Pipeline & Cross Validation              ////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create a CoxRegression object. This instance is an Estimator
    val cox = new CoxRegression(sc)
      .setFeaturesCol("dataset")
      .setLabelCol("dataset")
      .setFeatureIndicesRangeStart(1)
      .setFeatureIndicesRangeEnd(1000)
      .setTimestampIndex(1)
      .setEventIndex(2)              
    
    // Create the pipeline. Currently there is only one stage, i.e., Cox (Estimator)
    val pipeline = new Pipeline().setStages(Array(cox))
    
    // Cross Validation and Grid Search to find the best model.
    val crossval = new CrossValidator()
                       .setEstimator(pipeline)
                       .setEvaluator(new SurvivalAnalysisEvaluator().setLabelCol("dataset"))                       
    val paramGrid = new ParamGridBuilder()
                        .addGrid(cox.alpha, Array(0.05, 0.001))
                        .build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)
    val cvModel = crossval.fit(training)
    
    // Make predictions on test dataset and display the results.
    cvModel.transform(test).show()
  }
}