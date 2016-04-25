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

import org.scalatest.FunSuite
import org.scalatest.Matchers
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.linalg.Vector
import scala.reflect.runtime.universe._

case class LabeledDocument[T:TypeTag](id: Long, text: String, label: Double)
case class Document[T:TypeTag](id: Long, text: String)

class LogisticRegressionSuite extends FunSuite with WrapperSparkContext with Matchers with Logging {

  // Note: This is required by every test to ensure that it runs successfully on windows laptop !!!
  val loadConfig = ScalaAutomatedTestBase
  
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
    val lr = new LogisticRegression("log", sc)
    val lrmodel = lr.fit(training.toDF)
    lrmodel.transform(testing.toDF).show

    lr.getIcpt shouldBe 0
    lrmodel.getIcpt shouldBe lr.getIcpt
    lrmodel.getMaxInnerIter shouldBe lr.getMaxInnerIter
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

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val lr = new LogisticRegression("log",sc)
    val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
    val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator)
    val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lr.regParam, Array(0.1, 0.01)).build()
    crossval.setEstimatorParamMaps(paramGrid)
    crossval.setNumFolds(2)
    val lrmodel = crossval.fit(training.toDF)
    val test = sc.parallelize(Seq(
      Document(12L, "spark i j k"),
      Document(13L, "l m n"),
      Document(14L, "mapreduce spark"),
      Document(15L, "apache hadoop")))
    
    lrmodel.transform(test.toDF).show
    
    lr.getIcpt shouldBe 0
//    lrmodel.getIcpt shouldBe lr.getIcpt
//    lrmodel.getMaxInnerIter shouldBe lr.getMaxInnerIter
  }
}