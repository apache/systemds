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

import java.io.File

import akka.http.scaladsl.server.StandardRoute
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.model.StatusCodes
import akka.http.scaladsl.Http
import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import org.apache.commons.cli.PosixParser
import com.typesafe.config.ConfigFactory

import scala.concurrent.duration._
import java.util.HashMap

import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import spray.json._
import java.util.concurrent.atomic.LongAdder

import scala.concurrent.{Await, Future}
import scala.math.{max, pow}
import org.apache.sysml.runtime.matrix.data.{MatrixBlock, OutputInfo}
import org.apache.sysml.parser.DataExpression
import org.apache.sysml.runtime.io.IOUtilFunctions
import org.apache.sysml.api.jmlc.Connection
import org.apache.sysml.api.jmlc.PreparedScript
import org.apache.sysml.conf.ConfigurationManager
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.sysml.runtime.util.DataConverter
import org.apache.commons.logging.Log
import org.apache.commons.logging.LogFactory

import scala.concurrent.ExecutionContext

// format: can be file, binary, csv, ijv, jpeg, ...

case class RequestStatistics(var batchSize: Int = -1,
                             var execTime: Long = -1,
                             var execType: String = "",
                             var requestDeserializationTime: Long = -1,
                             var responseSerializationTime: Long = -1,
                             var modelAcquireTime: Long = -1,
                             var modelReleaseTime: Long = -1,
                             var batchingTime: Long = -1,
                             var unbatchingTime: Long = -1,
                             var queueWaitTime: Long = -1,
                             var queueSize: Int = -1,
                             var execMode: Int = 0,
                             var preprocWaitTime: Long = -1)
case class PredictionRequestExternal(name: String, data: Array[Double], rows: Int, cols: Int)
case class PredictionResponseExternal(response: Array[Double], rows: Int, cols: Int, statistics: RequestStatistics)

case class AddModelRequest(name: String, dml: String, inputVarName: String,
                           outputVarName: String, weightsDir: String,
                           latencyObjective: String, batchSize: Array[Int], memUse: Array[Long])

case class Model(name: String,
                 script: Map[String,PreparedScript],
                 inputVarName: String,
                 outputVarName: String,
                 latencyObjective: Duration,
                 weightFiles: Map[String, String],
                 coeffs: (Double, Double),
                 weightMem: Long)
case class PredictionRequest(data : MatrixBlock, modelName : String, requestSize : Int, receivedTime : Long)
case class PredictionResponse(response: MatrixBlock, batchSize: Int, statistics: RequestStatistics)
case class MatrixBlockContainer(numRows: Long, numCols: Long, nnz: Long, sum: Double, data: MatrixBlock)

trait PredictionJsonProtocol extends SprayJsonSupport with DefaultJsonProtocol {
    implicit val RequestStatisticsFormat = jsonFormat13(RequestStatistics)
    implicit val predictionRequestExternalFormat = jsonFormat4(PredictionRequestExternal)
    implicit val predictionResponseExternalFormat = jsonFormat4(PredictionResponseExternal)
}

trait AddModelJsonProtocol extends SprayJsonSupport with DefaultJsonProtocol {
    implicit val AddModelRequetFormat = jsonFormat8(AddModelRequest)
}

class PredictionService {

}

/*
Usage:
1. Compiling a fat jar with maven assembly plugin in our standalone jar created lot of issues. 
Hence, for time being, we recommend downloading jar using the below script:
SCALA_VERSION="2.11"
AKKA_HTTP_VERSION="10.1.3"
AKKA_VERSION="2.5.14"
PREFIX="http://central.maven.org/maven2/com/typesafe/akka/"
JARS=""
for PKG in actor stream protobuf
do
  PKG_NAME="akka-"$PKG"_"$SCALA_VERSION
  JAR_FILE=$PKG_NAME"-"$AKKA_VERSION".jar"
  wget $PREFIX$PKG_NAME"/"$AKKA_VERSION"/"$JAR_FILE
  JARS=$JARS$JAR_FILE":"
done
for PKG in http http-core parsing
do
  PKG_NAME="akka-"$PKG"_"$SCALA_VERSION
  JAR_FILE=$PKG_NAME"-"$AKKA_HTTP_VERSION".jar"
  wget $PREFIX$PKG_NAME"/"$AKKA_HTTP_VERSION"/"$JAR_FILE
  JARS=$JARS$JAR_FILE":"
done
wget http://central.maven.org/maven2/com/typesafe/config/1.3.3/config-1.3.3.jar
wget http://central.maven.org/maven2/com/typesafe/ssl-config-core_2.11/0.2.4/ssl-config-core_2.11-0.2.4.jar
wget http://central.maven.org/maven2/org/reactivestreams/reactive-streams/1.0.2/reactive-streams-1.0.2.jar
wget http://central.maven.org/maven2/org/scala-lang/scala-library/2.11.12/scala-library-2.11.12.jar
wget http://central.maven.org/maven2/org/scala-lang/scala-parser-combinators/2.11.0-M4/scala-parser-combinators-2.11.0-M4.jar
wget http://central.maven.org/maven2/commons-cli/commons-cli/1.4/commons-cli-1.4.jar
wget http://central.maven.org/maven2/com/typesafe/akka/akka-http-spray-json-experimental_2.11/2.4.11.2/akka-http-spray-json-experimental_2.11-2.4.11.2.jar
wget http://central.maven.org/maven2/io/spray/spray-json_2.11/1.3.2/spray-json_2.11-1.3.2.jar
JARS=$JARS"config-1.3.3.jar:ssl-config-core_2.11-0.2.4.jar:reactive-streams-1.0.2.jar:commons-cli-1.4.jar:scala-parser-combinators-2.11.0-M4.jar:scala-library-2.11.12.jar:akka-http-spray-json-experimental_2.11-2.4.11.2.jar:spray-json_2.11-1.3.2.jar"
echo "Include the following jars into the classpath: "$JARS


2. Copy SystemML.jar and systemml-1.2.0-SNAPSHOT-extra.jar into the directory where akka jars are placed

3. Start the server:
java -cp $JARS org.apache.sysml.api.ml.serving.PredictionService -port 9000 -admin_password admin

4. Check the health of the server:
curl -u admin -XGET localhost:9000/health

5. Perform prediction
curl -XPOST -H "Content-Type:application/json" -d '{ "inputs":"1,2,3", "format":"csv", "model":"test", "num_input":1 }' localhost:9000/predict

6. Shutdown the server:
curl -u admin -XGET localhost:9000/shutdown

 */

object PredictionService extends PredictionJsonProtocol with AddModelJsonProtocol {
    val __DEBUG__ = false

    val LOG = LogFactory.getLog(classOf[PredictionService].getName)
    val customConf = ConfigFactory.parseString("""
        akka.http.server.idle-timeout=infinite
        akka.http.client.idle-timeout=infinite
        akka.http.host-connection-pool.idle-timeout=infinite
        akka.http.host-connection-pool.client.idle-timeout=infinite
        akka.http.server.max-connections=100000
    """)
    val basicConf = ConfigFactory.load()
    val combined = customConf.withFallback(basicConf)
    implicit val system = ActorSystem("systemml-prediction-service", ConfigFactory.load(combined))
    implicit val materializer = ActorMaterializer()
    implicit val executionContext = ExecutionContext.global
    implicit val timeout = akka.util.Timeout(300.seconds)
    val userPassword = new HashMap[String, String]()
    var bindingFuture: Future[Http.ServerBinding] = null
    var scheduler: Scheduler = null
    val conn = new Connection()
    var existantMatrixBlocks = Array[MatrixBlockContainer]()

    def getCommandLineOptions(): org.apache.commons.cli.Options = {
        val hostOption = new org.apache.commons.cli.Option("ip", true, "IP address")
        val portOption = new org.apache.commons.cli.Option("port", true, "Port number")
        val numRequestOption = new org.apache.commons.cli.Option("max_requests", true, "Maximum number of requests")
        val timeoutOption = new org.apache.commons.cli.Option("timeout", true, "Timeout in milliseconds")
        val passwdOption = new org.apache.commons.cli.Option("admin_password", true, "Admin password. Default: admin")
        val helpOption = new org.apache.commons.cli.Option("help", false, "Show usage message")
        val maxSizeOption = new org.apache.commons.cli.Option("max_bytes", true, "Maximum size of request in bytes")
        val statisticsOption = new org.apache.commons.cli.Option("statistics", true, "Gather statistics on request execution")
        val numCpuOption = new org.apache.commons.cli.Option("num_cpus", true, "How many CPUs should be allocated to the prediction service. Default nproc-1")
        val gpusOption = new org.apache.commons.cli.Option("gpus", true, "GPUs available to this process. Default: 0")
        val schedulerOption = new org.apache.commons.cli.Option("scheduler", true, "Scheduler implementation to use. Default: locality-aware")

        // Only port is required option
        portOption.setRequired(true)

        return new org.apache.commons.cli.Options()
          .addOption(hostOption).addOption(portOption).addOption(numRequestOption)
          .addOption(passwdOption).addOption(timeoutOption).addOption(helpOption)
          .addOption(maxSizeOption).addOption(statisticsOption).addOption(numCpuOption)
          .addOption(gpusOption).addOption(schedulerOption)
    }

    def main(args: Array[String]): Unit = {
        // Parse commandline variables:
        val options = getCommandLineOptions
        val line = new PosixParser().parse(getCommandLineOptions, args)
        if (line.hasOption("help")) {
            new org.apache.commons.cli.HelpFormatter().printHelp("systemml-prediction-service", options)
            return
        }
        userPassword.put("admin", line.getOptionValue("admin_password", "admin"))
        val currNumRequests = new LongAdder
        val maxNumRequests = if (line.hasOption("max_requests"))
            line.getOptionValue("max_requests").toLong else Long.MaxValue
        val timeout = if (line.hasOption("timeout"))
            Duration(line.getOptionValue("timeout").toLong, MILLISECONDS) else 300.seconds
        val sizeDirective = if (line.hasOption("max_bytes"))
            withSizeLimit(line.getOptionValue("max_bytes").toLong) else withoutSizeLimit
        val numCores = if (line.hasOption("num_cpus"))
            line.getOptionValue("num_cpus").toInt else Runtime.getRuntime.availableProcessors() - 1
        val gpus = if (line.hasOption("gpus")) line.getOptionValue("gpus") else null
        val schedulerType = line.getOptionValue("scheduler", "locality-aware")

        // Initialize statistics counters
        val numTimeouts = new LongAdder
        val numFailures = new LongAdder
        val totalTime = new LongAdder
        val numCompletedPredictions = new LongAdder

        // For now the models need to be loaded every time. TODO: pass the local to serialized models via commandline
        var models = Map[String, Model]()

        // TODO: Set the scheduler using factory
        scheduler = SchedulerFactory.getScheduler(schedulerType)
        val maxMemory = Runtime.getRuntime.maxMemory()  // total memory is just what the JVM has currently allocated

        LOG.info("Total memory allocated to server: " + maxMemory)
        scheduler.start(numCores, maxMemory, gpus)

        // Define unsecured routes: /predict and /health
        val unsecuredRoutes = {
            path("predict") {
                withoutRequestTimeout {
                    post {
                        validate(currNumRequests.longValue() < maxNumRequests, "The prediction server received too many requests. Ignoring the current request.") {
                            entity(as[PredictionRequestExternal]) { request =>
                                validate(models.contains(request.name), "The model is not available.") {
                                    try {
                                        currNumRequests.increment()
                                        val start = System.nanoTime()
                                        val processedRequest = processPredictionRequest(request)
                                        val deserializationTime = System.nanoTime() - start

                                        val response = Await.result(
                                            scheduler.enqueue(processedRequest, models(request.name)), timeout)
                                        totalTime.add(System.nanoTime() - start)

                                        numCompletedPredictions.increment()
                                        complete(StatusCodes.OK, processPredictionResponse(response, "NOT IMPLEMENTED", deserializationTime))
                                    } catch {
                                        case e: scala.concurrent.TimeoutException => {
                                            numTimeouts.increment()
                                            complete(StatusCodes.RequestTimeout, "Timeout occured")
                                        }
                                        case e: Exception => {
                                            numFailures.increment()
                                            e.printStackTrace()
                                            val msg = "Exception occured while executing the prediction request:"
                                            complete(StatusCodes.InternalServerError, msg + e.getMessage)
                                        }
                                    } finally {
                                        currNumRequests.decrement()
                                    }
                                }
                            }
                        }
                    }
                }
            } ~ path("health") {
                get {
                    val stats = "Number of requests (total/completed/timeout/failures):" + currNumRequests.longValue() + "/" + numCompletedPredictions.longValue() + "/"
                    numTimeouts.longValue() + "/" + numFailures.longValue() + ".\n" +
                      "Average prediction time:" + ((totalTime.doubleValue() * 1e-6) / numCompletedPredictions.longValue()) + " ms.\n"
                    complete(StatusCodes.OK, stats)
                }
            }
        }

        // For administration: This can be later extended for supporting multiple users.
        val securedRoutes = {
            authenticateBasicAsync(realm = "secure site", userAuthenticate) {
                user =>
                    path("shutdown") {
                        get {
                            shutdownService(user, scheduler)
                        }
                    } ~
                      path("register-model") {
                          withoutRequestTimeout {
                              post {
                                  entity(as[AddModelRequest]) { request =>
                                      validate(!models.contains(request.name), "The model is already loaded") {
                                          try {
                                              val weightsInfo = processWeights(request.weightsDir)
                                              val inputs = weightsInfo._1.keys.toArray ++ Array[String](request.inputVarName)

                                              // compile for executor types
                                              val scriptCpu = conn.prepareScript(
                                                  request.dml, inputs, Array[String](request.outputVarName))
                                              var scripts = Map("CPU" -> scriptCpu)

                                              if (gpus != null) {
                                                  GPUContextPool.AVAILABLE_GPUS = gpus
                                                  for (ix <- 0 until GPUContextPool.getAvailableCount) {
                                                      LOG.info("Compiling script for GPU: " + ix)
                                                      scripts += (s"GPU${ix}" -> conn.prepareScript(
                                                          request.dml, inputs, Array[String](request.outputVarName),
                                                          true, true, ix))
                                                  }
                                              }

                                              // b = cov(x,y) / var(x)
                                              // a = mean(y) - b*mean(x)
                                              val n = max(request.batchSize.length, 1).toDouble
                                              val x = request.batchSize
                                              val y = request.memUse
                                              val mux = x.sum / n
                                              val muy = y.sum / n
                                              val vx = (1 / n) * x.map(v => pow(v - mux, 2.0)).sum
                                              val b = ((1 / n) * (x.map(v => v - mux) zip y.map(v => v - muy)
                                                ).map(v => v._1 * v._2).sum) * (1 / vx)
                                              val a = muy - b * mux

                                              // now register the created model
                                              val model = Model(request.name,
                                                  scripts,
                                                  request.inputVarName,
                                                  request.outputVarName,
                                                  Duration(request.latencyObjective),
                                                  weightsInfo._1, (a, b), weightsInfo._2)
                                              models += (request.name -> model)
                                              scheduler.addModel(model)
                                              complete(StatusCodes.OK)
                                          } catch {
                                              case e: Exception => {
                                                  numFailures.increment()
                                                  e.printStackTrace()
                                                  complete(StatusCodes.InternalServerError,
                                                      "Exception occured while trying to add model:" + e.getMessage)
                                              }
                                          }
                                      }
                                  }
                              }
                          }
                      }
            }
        }

        bindingFuture = Http().bindAndHandle(
            sizeDirective { // Both secured and unsecured routes need to respect the size restriction
                unsecuredRoutes ~ securedRoutes
            },
            line.getOptionValue("ip", "localhost"), line.getOptionValue("port").toInt)

        println(s"Prediction Server online.")
        while (true) Thread.sleep(100)
        bindingFuture
          .flatMap(_.unbind())
          .onComplete(_ ⇒ system.terminate())
    }

    def processPredictionResponse(response : PredictionResponse, 
                                  format : String, 
                                  deserializationTime: Long) : PredictionResponseExternal = {
        if (response != null) {
            val start = System.nanoTime()
            val dataArray = response.response.getDenseBlockValues
            val rows = response.response.getNumRows
            val cols = response.response.getNumColumns
            val serializationTime = System.nanoTime() - start
            if (response.statistics != null) {
                response.statistics.requestDeserializationTime = deserializationTime
                response.statistics.responseSerializationTime = serializationTime
            }
            PredictionResponseExternal(dataArray, rows, cols, response.statistics)
        } else {
            PredictionResponseExternal(null, -1, -1, null)
        }
    }

    def processWeights(dirname: String) : (Map[String, String], Long) = {
        val dir = new File(dirname)
        if (!(dir.exists && dir.isDirectory))
            throw new Exception("Weight directory: " + dirname + " is invalid")

        val weightsWithSize = dir.listFiles().filter(
            x => !(x.isDirectory && (x.toString contains "binary"))).map(_.toString).filter(
            x => (x.slice(x.length-3, x.length) != "mtd") &&
            !(x contains "_bin.mtx")).
          map(x => getNameFromPath(x) -> registerWeight(x, dirname)).toMap

        val weightMap = weightsWithSize.map(x => x._1 -> x._2._1)
        val totalSize = weightsWithSize.map(x => x._2._2).sum

        (weightMap, totalSize)
    }

    def getNameFromPath(path: String) : String = {
        path.split("/").last.split("\\.")(0)
    }

    def registerWeight(path: String, dir: String) : (String, Long) = {
        val res = convertToBinaryIfNecessary(path, dir)
        scheduler.modelManager.putWeight(res._2, res._1)
        (res._2, res._1.getInMemorySize)
    }

    def convertToBinaryIfNecessary(path: String, dir: String) : (MatrixBlock, String) = {
        var pathActual = path
        LOG.info("Reading weight: " + path)
        val data = conn.readMatrix(path)

        if (!isBinaryFormat(path)) {
            LOG.info("Converting weight to binary format")
            data.getMatrixCharacteristics
            val binPath = dir + "/binary/" + getNameFromPath(path) + ".mtx"
            DataConverter.writeMatrixToHDFS(data, binPath,
                OutputInfo.BinaryBlockOutputInfo,
                new MatrixCharacteristics(data.getNumRows, data.getNumColumns, ConfigurationManager.getBlocksize,
                    ConfigurationManager.getBlocksize, data.getNonZeros))
            pathActual = binPath
        }
        (data, pathActual)
    }

    def isBinaryFormat(path: String) : Boolean = {
        val mtdName = DataExpression.getMTDFileName(path)
        val mtd = new DataExpression().readMetadataFile(mtdName, false)
        if (mtd.containsKey("format")) mtd.getString("format") == "binary" else false
    }

    def processPredictionRequest(request : PredictionRequestExternal) : PredictionRequest = {
        val mat = new MatrixBlock(request.rows, request.cols, false)
        mat.init(request.data, request.rows, request.cols)
        PredictionRequest(mat, request.name, request.rows, System.nanoTime())
    }

    def processMatrixInput(data : String, rows : Int, cols : Int, format : String) : MatrixBlock = {
        val result = format match {
            case "csv" => processTextInput(data, rows, cols, DataExpression.FORMAT_TYPE_VALUE_CSV)
            case _ => throw new Exception("Only CSV Input currently supported")
        }
        result
    }

    def processTextInput(data : String, rows : Int, cols : Int, format : String) : MatrixBlock = {
        val is = IOUtilFunctions.toInputStream(data)
        conn.convertToMatrix(is, rows, cols, format)
    }

    def userAuthenticate(credentials: akka.http.scaladsl.server.directives.Credentials): Future[Option[String]] = {
        credentials match {
            case p@akka.http.scaladsl.server.directives.Credentials.Provided(id) =>
                Future {
                    if (userPassword.containsKey(id) && p.verify(userPassword.get(id))) Some(id)
                    else None
                }
            case _ => Future.successful(None)
        }
    }

    def shutdownService(user: String, scheduler: Scheduler): StandardRoute = {
        if (user.equals("admin")) {
            try {
                Http().shutdownAllConnectionPools() andThen { case _ => bindingFuture.flatMap(_.unbind()).onComplete(_ ⇒ system.terminate()) }
                scheduler.shutdown()
                complete(StatusCodes.OK, "Shutting down the server.")
            } finally {
                new Thread(new Runnable {
                    def run() {
                        Thread.sleep(100) // wait for 100ms to send reply and then kill the prediction JVM so that we don't wait scala.io.StdIn.readLine()
                        System.exit(0)
                    }
                }).start();
            }
        }
        else {
            complete(StatusCodes.BadRequest, "Only admin can shutdown the service.")
        }
    }

}