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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Statistics;

/**
 * TODO robustness on failures (cleanup files)
 * TODO heavy hitter maintenance
 * TODO data partitioning with binarycell
 *
 */
public class RemoteDPParForSpark
{
	
	protected static final Log LOG = LogFactory.getLog(RemoteDPParForSpark.class.getName());
	
	/**
	 * 
	 * @param pfid
	 * @param program
	 * @param taskFile
	 * @param resultFile
	 * @param enableCPCaching 
	 * @param mode
	 * @param numMappers
	 * @param replication
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static RemoteParForJobReturn runJob(long pfid, String itervar, String matrixvar, String program, String resultFile, MatrixObject input, 
			                                   ExecutionContext ec,
			                                   PDataPartitionFormat dpf, OutputInfo oi, boolean tSparseCol, //config params
			                                   boolean enableCPCaching, int numReducers )  //opt params
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		String jobname = "ParFor-DPESP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//prepare input parameters
		MatrixDimensionsMetaData md = (MatrixDimensionsMetaData) input.getMetaData();
		MatrixCharacteristics mc = md.getMatrixCharacteristics();
		InputInfo ii = InputInfo.BinaryBlockInputInfo;
				
		//initialize accumulators for tasks/iterations
		Accumulator<Integer> aTasks = sc.accumulator(0);
		Accumulator<Integer> aIters = sc.accumulator(0);
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
		DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(mc, ii, oi, dpf);
		RemoteDPParForSparkWorker efun = new RemoteDPParForSparkWorker(program, matrixvar, itervar, 
				          enableCPCaching, mc, tSparseCol, dpf, oi, aTasks, aIters);
		List<Tuple2<Long,String>> out = 
				in.flatMapToPair(dpfun)         //partition the input blocks
		          .groupByKey(numReducers)      //group partition blocks 		          
		          .mapPartitionsToPair( efun )  //execute parfor tasks, incl cleanup
		          .collect();                   //get output handles
		
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
