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

package org.apache.sysml.api.ml.util

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

object RDDConverterUtilsExt{
  /**
   * Add element indices column to input data frame 
   * Notes: This function dependent to RDD function [[RDD.zipWithIndex]]
   */
  def addIDToDataFrame(df:DataFrame, nameOfCol:String, start:Long):DataFrame = {
    import df.sqlContext.implicits._
    val indexDF = df.rdd.sparkContext.parallelize(0.toLong to df.count()).toDF(nameOfCol)
    val dataset1rdd = df.rdd.zipWithIndex().map(x=>Row((x._1.toSeq :+ (x._2 + start) ):_*))
    val schema = df.schema.fields :+ new StructField(nameOfCol, DataTypes.LongType, false)
    df.sqlContext.createDataFrame(dataset1rdd, new StructType(schema))
  }
}