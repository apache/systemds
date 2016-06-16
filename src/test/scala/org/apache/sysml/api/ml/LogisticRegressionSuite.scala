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

package org.apache.sysml.api.ml

import scala.reflect.runtime.universe

import org.apache.spark.Logging
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite
import org.scalatest.Matchers

case class LabeledDocument(id: Long, text: String, label: Double)
case class Document(id: Long, text: String)

class LogisticRegressionSuite extends FunSuite with WrapperSparkContext with Matchers with Logging {

  // Note: This is required by every test to ensure that it runs successfully on windows laptop !!!
  val loadConfig = ScalaAutomatedTestBase
  
  test("run logistic regression"){
    val newsqlContext = new org.apache.spark.sql.SQLContext(sc);

    import newsqlContext.implicits._
    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5))))
      //.map { x => LabeledPoint(x.label-1,x.features)}
    val testing = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5))))
      //.map { x => LabeledPoint(x.label-1,x.features)}
    val lr = new LogisticRegression("log")
    val lrmodel = lr.setTol(0.00003).fit(training.toDF)
    lrmodel.transform(testing.toDF).show
    
    lr.getTol shouldBe 0.00003
    lrmodel.getIcpt shouldBe lr.getIcpt
    lrmodel.getMaxInnerIter shouldBe lr.getMaxInnerIter    
  }
  
  test("run logistic regression with default") {
    //Make sure system ml home set when run wrapper
    val newsqlContext = new org.apache.spark.sql.SQLContext(sc);

    import newsqlContext.implicits._
    val training = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5))))
    val testing = sc.parallelize(Seq(
      LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
      LabeledPoint(1.0, Vectors.dense(1.0, 0.4, 2.1)),
      LabeledPoint(2.0, Vectors.dense(1.2, 0.0, 3.5))))
      
    //Systemml need label column as [1.0, 2.0]
    val lr_systemml = new LogisticRegression("log")
    val lrmodel_systemml = lr_systemml.fit(training.toDF)
    val scored_systemml = lrmodel_systemml.transform(testing.toDF).orderBy("ID")
    scored_systemml.select("prediction").collect() shouldEqual testing.toDF.select("label").collect()

    //Spark need label column as [0.0, 1.0]
    val lr_spark = new org.apache.spark.ml.classification.LogisticRegression("log")
    val lrmodel_spark = lr_spark.fit(training.map { x => LabeledPoint(x.label-1,x.features)}.toDF)
    val scored_spark = lrmodel_spark.transform(testing.map { x => LabeledPoint(x.label-1,x.features)}.toDF)
    scored_spark.select("prediction").collect() shouldEqual testing.map { x => LabeledPoint(x.label-1,x.features)}.toDF.select("label").collect()
    
    //scored_systemml.select("prediction").collect() shouldEqual scored_spark.select("prediction").collect()
  }

  test("test logistic regression with mlpipeline"){
    //Make sure system ml home set when run wrapper
    val newsqlContext = new org.apache.spark.sql.SQLContext(sc);
    import newsqlContext.implicits._
    val training = sc.parallelize(Seq(
	     LabeledDocument(0L, "a b c d e spark", 1.0),
	     LabeledDocument(1L, "b d", 2.0),
	     LabeledDocument(2L, "spark f g h", 1.0),
	     LabeledDocument(3L, "hadoop mapreduce", 2.0),
	     LabeledDocument(4L, "b spark who", 1.0),
	     LabeledDocument(5L, "g d a y", 2.0),
	     LabeledDocument(6L, "spark fly", 1.0),
	     LabeledDocument(7L, "was mapreduce", 2.0),
	     LabeledDocument(8L, "e spark program", 1.0),
	     LabeledDocument(9L, "a e c l", 2.0),
	     LabeledDocument(10L, "spark compile", 1.0),
	     LabeledDocument(11L, "hadoop software", 2.0)))
	     
    val test = sc.parallelize(Seq(
      Document(12L, "spark i j k"),
      Document(13L, "l m n"),
      Document(14L, "mapreduce spark"),
      Document(15L, "apache hadoop")))
      
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    
    
    val lr_systemml = new LogisticRegression("log")
    val pipeline_systemml = new Pipeline().setStages(Array(tokenizer, hashingTF, lr_systemml))
    val crossval_systemml = new CrossValidator().setEstimator(pipeline_systemml).setEvaluator(new BinaryClassificationEvaluator)
    val paramGrid_systemml = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr_systemml.regParam, Array(0.1, 0.01)).build()
    crossval_systemml.setEstimatorParamMaps(paramGrid_systemml)
    crossval_systemml.setNumFolds(2)
    val model_systemml = crossval_systemml.fit(training.toDF)
    val scored_systemml = model_systemml.transform(test.toDF).orderBy("id")
    
    val lr_spark = new org.apache.spark.ml.classification.LogisticRegression("log")
    val pipeline_spark = new Pipeline().setStages(Array(tokenizer, hashingTF, lr_spark))
    val crossval_spark = new CrossValidator().setEstimator(pipeline_spark).setEvaluator(new BinaryClassificationEvaluator)
    val paramGrid_spark = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr_systemml.regParam, Array(0.1, 0.01)).build()
    crossval_spark.setEstimatorParamMaps(paramGrid_spark)
    crossval_spark.setNumFolds(2)
    val model_spark = crossval_spark.fit(training.map { x => LabeledDocument(x.id,x.text, x.label-1)}.toDF)
    val scored_spark = model_spark.transform(test.toDF).orderBy("id")
    
    scored_systemml.select("prediction").collect() shouldEqual scored_spark.select(scored_spark("prediction")+1).collect()
  }
}