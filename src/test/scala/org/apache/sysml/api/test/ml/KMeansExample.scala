package org.apache.sysml.api.test.ml

import org.apache.sysml.api.ml.clustering.KMeans
import java.io.File
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext

object KMeansExample {
  def main(args: Array[String]) {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    
    val sparkConf = new SparkConf()
                .setAppName("KMeans Exanple ")
                .setMaster("local");
    
    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    
    import sqlContext.implicits._
    
//    val fPath = this.getClass().getResource("/KMeans_X.mtx").getPath()
    val fPath = "src" + File.separator + "test" + File.separator + "resources" + File.separator + "KMeans_X.mtx";

    val x = sc.textFile(fPath)
    val dataset = x.zipWithIndex().map { case (line, i) => (i + 1, Vectors.dense(line.split(",").map(_.toDouble))) }.toDF("__INDEX", "features")
    
    val km = new KMeans(sc).setK(5)
    
    val pipeline = new Pipeline().setStages(Array(km))
    val model = pipeline.fit(dataset)
//    val result = model.transform(dataset)
    
//    result.show()
  }
}
