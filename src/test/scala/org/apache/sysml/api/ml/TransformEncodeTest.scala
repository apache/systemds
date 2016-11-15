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

import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._

import org.apache.spark.sql._
import org.apache.spark.sql.types.{StructType,StructField,DoubleType,StringType,IntegerType}

import org.scalatest.FunSuite
import org.scalatest.Matchers
import org.apache.spark.Logging


class TransformEncodeTest extends FunSuite with WrapperSparkContext with Matchers with Logging {

  // Note: This is required by every test to ensure that it runs successfully on windows laptop !!!
  val loadConfig = ScalaAutomatedTestBase
  
  test("run logistic regression with default") {

    val dataA = sc. parallelize( Array( Row("Doc1", "Feat1", 10), Row("Doc1", "Feat2", 20), Row("Doc2", "Feat1", 31)))
    val dataB = sc. parallelize( Array( Row("Doc1", "cat"), Row("Doc2", "dog")))
    val dataC = sc. parallelize( Array( Row(10.0, 2.0), Row(20.0, 1.0)))
    val schemaA = StructType( Array( StructField("myID", StringType, true), StructField("FeatureName", StringType, true), StructField("FeatureValue", IntegerType, true)) )
    val schemaB = StructType( Array( StructField("myID", StringType, true), StructField("Label", StringType, true)) )
    val schemaC = StructType( Array( StructField("myID", DoubleType, true), StructField("Score", DoubleType, true)) )
    val A = sqlContext.createDataFrame(dataA, schemaA)
    val B = sqlContext.createDataFrame(dataB, schemaB)
    val C = sqlContext.createDataFrame(dataC, schemaC)
    
    val ml = new MLContext(sc)
    
    // ml.setExplain(true)
    
    //    A = read($A, datatype="frame", format="csv")
    //    B = read($B, datatype="frame", format="csv")
    //    C = read($C, datatype="frame", format="csv")
    //    write(tB, "tmp/tB.matrix.cvs", format="csv")
    //    write(tC, "tmp/tC.frame.cvs", format="csv")
    
    val dmlScript = """
        X = A[,1:2]
        [tA, tAM] = transformencode (target=X, spec="{ids:false, recode:[ myID,  FeatureName]}") 
        tB1 = transformapply( target=B[,1], spec="{ids:false, recode:[myID]}", meta=tAM)
        [tB2, tBM] = transformencode( target=B[,2], spec="{ids:false, recode:[Label]}")
        tB = append( tB1, tB2)
        tC = transformdecode( target=C[,2], spec="{ids:true, recode:[1]}", meta=tBM)
    """
//        tB1 = transformapply( target=B[,1], spec="{ids:false, recode:[myID]}", meta=tAM)
//        [tB2, tBM] = transformencode( target=B[,2], spec="{ids:false, recode:[Label]}")
//        tB = append( tB1, tB2)
//        tC = transformdecode( target=C[,2], spec="{ids:true, recode:[1]}", meta=tBM)
//    """
    
    //val dmlScript = "X = A[,1:2]; [tA, tAM] = transformencode (target=X, spec="{ids:false, recode:[ myID ]}"); tB1 = transformapply( target=B[,1], spec="{ids:false, recode:[myID]}", meta=tAM); [tB2, tBM] = transformencode( target=B[,2], spec="{ids:false, recode:[Label]}"); tB = append( tB1, tB2); tC = transformdecode( target=C[,2], spec="{ids:true, recode:[1]}", meta=tBM);"
    
    
    
    val dmlProg = dml(dmlScript)
                        .in("A", A)
                        .in("B", B)
                        .in("C", C)
                        .out("tA", "tB", "tAM", "tBM", "tC", "C")
    val dmlRes = ml.execute(dmlProg)
    
    val tA = dmlRes.getMatrix("tA").toDF.show
    val tAM = dmlRes.getFrame("tAM").toDF.show
    val tB = dmlRes.getMatrix("tB").toDF.show
    val tBM = dmlRes.getFrame("tBM").toDF.show
    val tC = dmlRes.getFrame("tC").toDF.show
    val oC = dmlRes.getMatrix("C").toDF.show
    
  }
}
