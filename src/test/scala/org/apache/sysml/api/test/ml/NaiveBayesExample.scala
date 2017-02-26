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

import org.apache.sysml.api.ml.classification.NaiveBayes
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.SQLContext

// case class LabeledDocument(id: Long, text: String, label: Double)
// case class Document(id: Long, text: String)

object NaiveBayesExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val sparkConf = new SparkConf()
                .setAppName("Naive Bayes")
                .setMaster("local");
    
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    val training = sqlContext.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 2.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 2.0),
      (4L, "b spark who", 1.0),
      (5L, "g d a y", 2.0),
      (6L, "spark fly", 1.0),
      (7L, "was mapreduce", 2.0),
      (8L, "e spark program", 1.0),
      (9L, "a e c l", 2.0),
      (10L, "spark compile", 1.0),
      (11L, "hadoop software", 2.0)
    )).toDF("id", "text", "label")
    
    val tokenizer = new Tokenizer()
                        .setInputCol("text")
                        .setOutputCol("words")
    val hashingTF = new HashingTF()
                        .setNumFeatures(1000)
                        .setInputCol(tokenizer.getOutputCol)
                        .setOutputCol("features")
    val nb = new NaiveBayes(sc)
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, nb))
    val crossval = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
    val paramGrid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures, Array(100))
      .addGrid(nb.laplaceCorrection, Array(1, 2)).build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)
    val cvModel = crossval.fit(training)
    
    val test = sqlContext.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "mapreduce spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")
    
    cvModel.transform(test).show()
  }
}