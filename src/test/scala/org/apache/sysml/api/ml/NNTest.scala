package org.apache.sysml.api.ml

import org.apache.spark.sql.SparkSession
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext._
import org.scalatest.FunSuite

object NNTest extends FunSuite with WrapperSparkContext {

  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder().master("local[4]").appName("NNTest").getOrCreate()
    val ml = new MLContext(sc)
    ml.setConfigProperty("sysml.native.blas.directory", "/opt/OpenBLAS/lib")
    ml.setConfigProperty("sysml.native.blas", "openblas")

    val trainScript = dmlFromFile("src/test/scripts/nn/cnn_test.dml")
      .in(Map("$images" -> "/media/bobo/Work/gsoc/data/train_features", "$labels" -> "/media/bobo/Work/gsoc/data/train_labels", "C" -> 1, "Hin" -> 28, "Win" -> 28, "epochs" -> 2))
      .out("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")
    val (w1, b1, w2, b2, w3, b3, w4, b4) = ml.execute(trainScript).getTuple[Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix, Matrix]("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4")

//    val testScript = dmlFromFile("src/test/scripts/nn/nn_test.dml")
//      .in(Map("$X_test" -> "/media/bobo/Work/gsoc/data/val_features", "$y_test" -> "/media/bobo/Work/gsoc/data/val_labels", "C" -> 1, "Hin" -> 28, "Win" -> 28, "W1" -> w1, "b1" -> b1, "W2" -> w2, "b2" -> b2, "W3" -> w3, "b3" -> b3, "W4" -> w4, "b4" -> b4))
//    ml.execute(testScript)

  }
}
