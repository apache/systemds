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

package org.apache.sysml.api.ml.scala

import org.scalatest.FunSuite
import org.scalatest.Matchers
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

class LogisticRegressionSuite extends FunSuite with WrapperSparkContext with Matchers with Logging{

  test("run logistic regression with default"){
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
    val lr = new LogisticRegression("log",sc)
    val lrmodel = lr.fit(training.toDF)
    lrmodel.transform(testing.toDF).show
    
    lr.getIcpt shouldBe 0
    lrmodel.getIcpt shouldBe lr.getIcpt
    lrmodel.getMaxInnerIter shouldBe lr.getMaxInnerIter
  }
}