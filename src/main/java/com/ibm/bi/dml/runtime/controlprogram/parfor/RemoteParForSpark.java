/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.utils.Statistics;

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
 * TODO broadcast variables if possible
 * TODO reducebykey on variable names
 */
public class RemoteParForSpark 
{
	
	protected static final Log LOG = LogFactory.getLog(RemoteParForSpark.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param tasks
	 * @param ec
	 * @param enableCPCaching
	 * @param numMappers
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static RemoteParForJobReturn runJob(long pfid, String program, List<Task> tasks, ExecutionContext ec,
			                                   boolean cpCaching, int numMappers) 
		throws DMLRuntimeException, DMLUnsupportedOperationException  
	{
		String jobname = "ParFor-ESP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//initialize accumulators for tasks/iterations
		Accumulator<Integer> aTasks = sc.accumulator(0);
		Accumulator<Integer> aIters = sc.accumulator(0);
		
		//run remote_spark parfor job 
		//(w/o lazy evaluation to fit existing parfor framework, e.g., result merge)
		RemoteParForSparkWorker func = new RemoteParForSparkWorker(program, cpCaching, aTasks, aIters);
		List<Tuple2<Long,String>> out = 
				sc.parallelize( tasks, numMappers )  //create rdd of parfor tasks
		          .flatMapToPair( func )             //execute parfor tasks 
		          .collect();                        //get output handles
		
		//de-serialize results
		LocalVariableMap[] results = RemoteParForUtils.getResults(out, LOG);
		int numTasks = aTasks.value(); //get accumulator value
		int numIters = aIters.value(); //get accumulator value
		
		//create output symbol table entries
		RemoteParForJobReturn ret = new RemoteParForJobReturn(true, numTasks, numIters, results);
		
		//maintain statistics
	    Statistics.incrementNoOfCompiledSPInst();
	    Statistics.incrementNoOfExecutedSPInst();
	    if( DMLScript.STATISTICS ){
			Statistics.maintainCPHeavyHitters(jobname, System.nanoTime()-t0);
		}
		
		return ret;
	}
}
