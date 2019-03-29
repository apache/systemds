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

import java.util
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.LongAdder

import org.apache.commons.logging.{Log, LogFactory}
import org.apache.sysml.api.jmlc.{Connection, PreparedScript}
import org.apache.sysml.runtime.matrix.data.MatrixBlock
import org.apache.sysml.utils.PersistentLRUCache

trait ModelManager {
    val LOG: Log = LogFactory.getLog(classOf[ModelManager].getName)
    var modelLocality = new ConcurrentHashMap[String, util.ArrayList[JmlcExecutor]]()
    val conn: Connection = new Connection()
    val availableMemory = new LongAdder
    var totalMemory = 0L
    var cleanupEnabled = true
    var memCheckEnabled = true
    var models: Map[String, Model] = Map()

    def setAvailableMemory(memBytes: Long) : Unit = {
        LOG.info("Setting total memory to: " + memBytes + " bytes")
        totalMemory = memBytes
        availableMemory.reset()
        availableMemory.add(memBytes)
    }

    def getAvailableMemory : Long = { availableMemory.longValue() }

    def acquireMemory(bytes: Long) : Long = {
        // if memory checking is not enabled just always say they get the memory
        if (!memCheckEnabled || bytes == 0)
            return bytes
        LOG.debug("Requested: " + bytes)

        // otherwise check to see if there is enough memory to meet the request
        if (bytes <= availableMemory.longValue()) {
            availableMemory.add(-1 * bytes)
            LOG.debug("Granted: " + bytes + "/" + availableMemory.longValue())
            return bytes
        }
        // not enough memory available :(

        LOG.debug("Insufficient memory. Request was not granted")
        -1
    }

    def releaseMemory(bytes: Long) : Unit = {
        if (bytes > 0) {
            LOG.debug("Releasing: " + bytes)
            availableMemory.add(bytes)
            LOG.debug("Available memory is now: " + availableMemory.longValue())
        }
    }

    def setModelLocality(model: String, exec: JmlcExecutor) : Unit = {
        this.synchronized({
            modelLocality.putIfAbsent(model, new util.ArrayList[JmlcExecutor]())
            modelLocality.get(model).add(exec)
        })
    }

    def unsetModelLocality(model: String, exec: JmlcExecutor) : Unit = {
        this.synchronized({ modelLocality.get(model).remove(exec) })
    }

    def getModelLocality(model: String) : util.ArrayList[JmlcExecutor] = { modelLocality.get(model) }

    def isModelLocal(model: String, exec: JmlcExecutor) : Boolean = { getModelLocality(model).contains(exec) }

    def disableCleanup() : Unit = { cleanupEnabled = false }

    def disableMemcheck() : Unit = { memCheckEnabled = false }

    def put(model: Model): Unit

    def get(name: String): Model

    def putWeight(name: String, weight: MatrixBlock) : Unit

    def acquire(name: String, executor: JmlcExecutor) : PreparedScript

    def release(name: String) : Unit
}

object ReferenceCountedModelManager extends ModelManager {
    var modelRefCounts: Map[String,LongAdder] = Map()
    var weightCache : PersistentLRUCache = _

    override def setAvailableMemory(maxBytes: Long) : Unit = {
        super.setAvailableMemory(maxBytes)
        weightCache = new PersistentLRUCache((0.80*maxBytes).toLong)
        weightCache.enableReadOnlyMode(true)
    }

    def tryAllocMem(name: String, batchSize: Int) : Long = {
        // TODO: More sophisticated memory management
        val extraMem = (0.5*models(name).weightMem).toLong
        val weightMem = if (modelRefCounts(name).longValue() > 0) 0L else models(name).weightMem
        val memReceived = acquireMemory(extraMem + weightMem)
        if (memReceived < 0) memReceived else extraMem
    }

    def isCached(name: String) : Boolean = { modelRefCounts(name).longValue() > 0 }

    def acquire(name: String, executor: JmlcExecutor) : PreparedScript = {
         LOG.debug("Acquiring model: " + name + " Ref count: " + modelRefCounts(name).longValue())

        val execName = if (executor.getExecType == "GPU") executor.getName else executor.getExecType
        val ps = models(name).script(execName)
        if (modelRefCounts(name).longValue() > 0 && ps.hasPinnedData) {
            modelRefCounts(name).increment()
            return ps.clone(false)
        }

        // otherwise we need to re-pin the weights, possibly reading them from disk
        val model = models(name)
        model.synchronized {
            LOG.debug("Pinning weights for: " + name)
            model.weightFiles.foreach(x => ps.setMatrix(x._1, weightCache.getAsMatrixBlock(x._2), true))
            modelRefCounts(name).increment()
        }
        LOG.debug("Done acquiring model: " + name)
        ps.clone(false)
    }

    override def disableCleanup(): Unit = {
        super.disableCleanup()
        LOG.debug("Cleanup is disabled")
    }

    def release(name: String) : Unit = {
        modelRefCounts(name).decrement()
        releaseMemory(models(name).weightMem)

        LOG.debug("Releasing model: " + name + " Ref count: " + modelRefCounts(name).longValue())
        if (modelRefCounts(name).longValue() == 0) {
            models(name).script.synchronized {
                if (modelRefCounts(name).longValue() == 0) {
                    models(name).script.foreach { x => x._2.clearPinnedData() }
                }
            }
        }
    }

    def put(model: Model) : Unit = {
        models += (model.name -> model)
        modelRefCounts += (model.name -> new LongAdder())
    }

    def putWeight(name: String, weight: MatrixBlock) : Unit = {
        weightCache.put(name, weight)
    }

    def get(name: String) : Model = { models(name) }

}