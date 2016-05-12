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