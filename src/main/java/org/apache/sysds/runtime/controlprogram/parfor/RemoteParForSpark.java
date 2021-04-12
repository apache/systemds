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

package org.apache.sysds.runtime.controlprogram.parfor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.ParForStatementBlock.ResultVar;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.utils.Statistics;

import scala.Tuple2;

/**
 * This class serves two purposes: (1) isolating Spark imports to enable running in 
 * environments where no Spark libraries are available, and (2) to follow the same
 * structure as the parfor remote_mr job submission.
 * 
 * NOTE: currently, we still exchange inputs and outputs via hdfs (this covers the general case
 * if data already resides in HDFS, in-memory data, and partitioned inputs; also, it allows for
 * pre-aggregation by overwriting partial task results with pre-paggregated results from subsequent
 * iterations)
 * 
 * TODO reducebykey on variable names
 */
public class RemoteParForSpark 
{
	protected static final Log LOG = LogFactory.getLog(RemoteParForSpark.class.getName());
	
	//globally unique id for parfor spark job instances (unique across spark contexts)
	private static final IDSequence _jobID = new IDSequence();
	
	public static RemoteParForJobReturn runJob(long pfid, String prog, HashMap<String, byte[]> clsMap, List<Task> tasks,
		ExecutionContext ec, Set<String> brVars, List<ResultVar> resultVars, boolean cpCaching, int numMappers, boolean topLevelPF)
	{
		String jobname = "ParFor-ESP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//initialize accumulators for tasks/iterations
		LongAccumulator aTasks = sc.sc().longAccumulator("tasks");
		LongAccumulator aIters = sc.sc().longAccumulator("iterations");
		
		//reset cached shared inputs for correctness in local mode
		long jobid = _jobID.getNextID();
		if( InfrastructureAnalyzer.isLocalMode() )
			RemoteParForSparkWorker.cleanupCachedVariables(jobid);

		// broadcast the inputs except the result variables
		Map<String, Broadcast<CacheBlock>> brInputs = null;
		if (ParForProgramBlock.ALLOW_BROADCAST_INPUTS)
			brInputs = broadcastInputs(sec, brVars);
		
		//prepare lineage
		Map<String, String> serialLineage = DMLScript.LINEAGE ? ec.getLineage().serialize() : null;

		//run remote_spark parfor job 
		//(w/o lazy evaluation to fit existing parfor framework, e.g., result merge)
		List<Tuple2<Long, String>> out = sc.parallelize(tasks, tasks.size()) //create rdd of parfor tasks
			.flatMapToPair(new RemoteParForSparkWorker(jobid, prog,
				clsMap, cpCaching, aTasks, aIters, brInputs, topLevelPF, serialLineage))
			.collect(); //execute and get output handles
		
		//de-serialize results
		LocalVariableMap[] results = RemoteParForUtils.getResults(out, LOG);
		Lineage[] lineages = DMLScript.LINEAGE ?
			RemoteParForUtils.getLineages(results) : null;
		
		int numTasks = aTasks.value().intValue(); //get accumulator value
		int numIters = aIters.value().intValue(); //get accumulator value
		
		//create output symbol table entries
		RemoteParForJobReturn ret = new RemoteParForJobReturn(true, numTasks, numIters, results, lineages);
		
		//maintain statistics
		Statistics.incrementNoOfCompiledSPInst();
		Statistics.incrementNoOfExecutedSPInst();
		if( DMLScript.STATISTICS )
			Statistics.maintainCPHeavyHitters(jobname, System.nanoTime()-t0);
		
		return ret;
	}

	@SuppressWarnings("unchecked")
	private static Map<String, Broadcast<CacheBlock>> broadcastInputs(SparkExecutionContext sec, Set<String> brVars) {
		// construct broadcast objects
		Map<String, Broadcast<CacheBlock>> result = new HashMap<>();
		for (String key : brVars) {
			Data var = sec.getVariable(key);
			if ((var instanceof ScalarObject) || (var instanceof MatrixObject && ((MatrixObject) var).isPartitioned()))
				continue;
			result.put(key, sec.broadcastVariable((CacheableData<CacheBlock>) var));
		}
		return result;
	}
}
