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
package org.apache.sysml.api.ml.serving

import java.util.concurrent.LinkedBlockingQueue

import breeze.linalg._
import breeze.numerics._
import breeze.stats._

class RLSEstimator {
    val dataQueue = new LinkedBlockingQueue[(Double, Double)]()
    val chunkSize = 2

    var isInitialized = false
    var isFinalized = false
    var Q : DenseMatrix[Double] = _
    var b : DenseMatrix[Double] = _
    var n = 0
    val lda = 0.98
    val eps = 0.00000001
    var sigma = -1.0

    def enqueueExample(batchSize: Int, latency: Double) : Unit = {
        if (!isFinalized) {
            println("ENQUEUING => " + dataQueue.size())
            dataQueue.add((batchSize.toDouble, latency))
            if (dataQueue.size() >= chunkSize)
                update()
        }
    }

    def dequeueExamples() : (DenseMatrix[Double], DenseMatrix[Double]) = {
        val X = DenseMatrix.zeros[Double](chunkSize,4)
        val y = DenseMatrix.zeros[Double](chunkSize, 1)

        for (ix <- 0 until chunkSize) {
            val (x_ex, y_ex) = dataQueue.poll()
            X(ix,::) := DenseVector[Double](1.0, x_ex, pow(x_ex,2), pow(x_ex,3)).t
            y(ix,0) = y_ex
        }
        (X, y)
    }

    def update() : Unit = {
        val s = pow(lda, n)
        val R = dequeueExamples()
        val X = R._1
        val y = R._2
        if (!isInitialized) {
            Q = X.t * X
            b = Q \ (X.t * y)
            isInitialized = true
        } else if (s >= eps) {
            val Q_new = Q + (X.t * X)
            val S = pow(lda, n) * DenseMatrix.eye[Double](chunkSize)
            val K = inv(Q_new) * (X.t * S) // Kalman filter gain
            val V = y - (X * b) // Innovations
            b :+= K * V
            Q = Q_new
        } else {
            isFinalized = true
            dataQueue.clear()
        }
        sigma = variance(y - (X*b))
        n += 1
    }

    def predict(batchSize: Int) : (Double,Double) = {
        val x = DenseMatrix(1.0, batchSize, pow(batchSize,2), pow(batchSize,3)).reshape(1,4)
        val y_hat = x*b
        (max(y_hat(0,0), 0.0), sigma)
    }

}