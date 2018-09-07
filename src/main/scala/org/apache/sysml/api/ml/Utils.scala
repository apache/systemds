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

import org.apache.spark.api.java.JavaPairRDD
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

object Utils {
  val originalOut = System.out
  val originalErr = System.err
}
class Utils {
  def checkIfFileExists(filePath: String): Boolean =
    return org.apache.sysml.runtime.util.MapReduceTool.existsFileOnHDFS(filePath)

  // --------------------------------------------------------------------------------
  // Simple utility function to print the information about our binary blocked format
  def getBinaryBlockInfo(binaryBlocks: JavaPairRDD[MatrixIndexes, MatrixBlock]): String = {
    val sb             = new StringBuilder
    var partitionIndex = 0
    for (str <- binaryBlocks.rdd.mapPartitions(binaryBlockIteratorToString(_), true).collect) {
      sb.append("-------------------------------------\n")
      sb.append("Partition " + partitionIndex + ":\n")
      sb.append(str)
      partitionIndex = partitionIndex + 1
    }
    sb.append("-------------------------------------\n")
    return sb.toString()
  }
  def binaryBlockIteratorToString(it: Iterator[(MatrixIndexes, MatrixBlock)]): Iterator[String] = {
    val sb = new StringBuilder
    for (entry <- it) {
      val mi = entry._1
      val mb = entry._2
      sb.append(mi.toString);
      sb.append(" sparse? = ");
      sb.append(mb.isInSparseFormat());
      if (mb.isUltraSparse)
        sb.append(" (ultra-sparse)")
      sb.append(", nonzeros = ");
      sb.append(mb.getNonZeros);
      sb.append(", dimensions = ");
      sb.append(mb.getNumRows);
      sb.append(" X ");
      sb.append(mb.getNumColumns);
      sb.append("\n");
    }
    List[String](sb.toString).iterator
  }
  val baos = new java.io.ByteArrayOutputStream()
  val baes = new java.io.ByteArrayOutputStream()
  def startRedirectStdOut(): Unit = {
    System.setOut(new java.io.PrintStream(baos));
    System.setErr(new java.io.PrintStream(baes));
  }
  def flushStdOut(): String = {
    val ret = baos.toString() + baes.toString()
    baos.reset(); baes.reset()
    return ret
  }
  def stopRedirectStdOut(): String = {
    val ret = baos.toString() + baes.toString()
    System.setOut(Utils.originalOut)
    System.setErr(Utils.originalErr)
    return ret
  }
  // --------------------------------------------------------------------------------
}
