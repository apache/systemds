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

import java.util.concurrent.{ConcurrentHashMap, CountDownLatch}

import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.math.min

object BasicBatchingScheduler extends BatchingScheduler {

    override def start(numCores: Int, cpuMemoryBudgetInBytes: Long, gpus: String): Unit = {
        LOG.info(s"Starting Basic Batching Scheduler with: ${numCores} CPUs and ${gpus} GPUs")
        super.start(numCores, cpuMemoryBudgetInBytes, gpus)
    }

    /**
      * Returns a list of requests to execute. If the list contains more than one element, they will be batched
      * by the executor. Returns an empty list when there are no models to be scheduled.
      * @param executor an Executor instance
      * @return a list of model requests to process
      */
    override def schedule(executor: JmlcExecutor) : Array[SchedulingRequest] = {
        var ret = Array[SchedulingRequest]()
        val execType = executor.getExecType
        dummyResponse.synchronized {
            val schedulableModels = getSchedulableModels(execType)
            if (schedulableModels.nonEmpty) {
                val (nextModel, nextBatchSize) = getNextModelAndBatchSize(schedulableModels, execType)
                for (_ <- 0 until nextBatchSize) {
                    val next = modelQueues.get(nextModel).poll()
                    assert(next != null, "Something is wrong. Next model should not be null")
                    ret :+= next
                }
            }
        }
        ret
    }

    /**
      * Helper method which gets the next model to schedule and the optimal batchsize
      * @param models A list of models to schedule
      * @return The model to schedule next
      */
    def getNextModelAndBatchSize(models : Iterable[String], execType: String) : (String, Int) = {
        val nextModel = models.map(m =>
            (getOptimalBatchSize(m, execType)*getExpectedExecutionTime(m), m)).minBy(x => x._1)._2

        val nextBatchSize = min(modelQueues.get(nextModel).size(),
            getOptimalBatchSize(nextModel, execType))
        (nextModel, nextBatchSize)
    }

    /**
      * Enqueues a request for processing. The scheduler will read from these queues to determine which
      * models to execute next
      * @param request A PredictionRequest object containing the data for which a prediction is desired
      * @param model The model object for which prediction
      * @return
      */
    override private[serving] def enqueue(request: PredictionRequest, model: Model): Future[PredictionResponse] = Future {
        val statistics = if (_statistics) RequestStatistics() else null
        val schedulingRequest = SchedulingRequest(
            request, model, new CountDownLatch(1), System.nanoTime(), null, statistics)
        statistics.queueSize = modelQueues.get(model.name).size
        modelQueues.get(model.name).add(schedulingRequest)
        counter += 1
        try {
            schedulingRequest.latch.await(timeout.length, timeout.unit)
            schedulingRequest.response
        } catch {
            case e : scala.concurrent.TimeoutException => dummyResponse
        }
    }

}