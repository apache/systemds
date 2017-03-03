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

package org.apache.sysml.runtime.controlprogram.parfor;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.util.LongAccumulator;

import scala.Tuple2;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.SparkUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixDimensionsMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.utils.Statistics;

/**
 * TODO heavy hitter maintenance
 * TODO data partitioning with binarycell
 *
 */
public class RemoteDPParForSpark
{
	
	protected static final Log LOG = LogFactory.getLog(RemoteDPParForSpark.class.getName());

	public static RemoteParForJobReturn runJob(long pfid, String itervar, String matrixvar, String program, HashMap<String, byte[]> clsMap,
			String resultFile, MatrixObject input, ExecutionContext ec, PDataPartitionFormat dpf, OutputInfo oi, 
			boolean tSparseCol, boolean enableCPCaching, int numReducers ) 
		throws DMLRuntimeException
	{
		String jobname = "ParFor-DPESP";
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//prepare input parameters
		MatrixDimensionsMetaData md = (MatrixDimensionsMetaData) input.getMetaData();
		MatrixCharacteristics mc = md.getMatrixCharacteristics();
		InputInfo ii = InputInfo.BinaryBlockInputInfo;

		//initialize accumulators for tasks/iterations, and inputs
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
		LongAccumulator aTasks = sc.sc().longAccumulator("tasks");
		LongAccumulator aIters = sc.sc().longAccumulator("iterations");

		//compute number of reducers (to avoid OOMs and reduce memory pressure)
		int numParts = SparkUtils.getNumPreferredPartitions(mc, in);
		int numParts2 = (int)((dpf==PDataPartitionFormat.ROW_BLOCK_WISE) ? mc.getRows() : mc.getCols()); 
		int numReducers2 = Math.max(numReducers, Math.min(numParts, numParts2));
		
		//core parfor datapartition-execute (w/ or w/o shuffle, depending on data characteristics)
		DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(mc, ii, oi, dpf);
		RemoteDPParForSparkWorker efun = new RemoteDPParForSparkWorker(program, clsMap, 
				matrixvar, itervar, enableCPCaching, mc, tSparseCol, dpf, oi, aTasks, aIters);
		JavaPairRDD<Long,Writable> tmp = in.flatMapToPair(dpfun);
		List<Tuple2<Long,String>> out = (requiresGrouping(dpf, mc) ?
				tmp.groupByKey(numReducers2) : tmp.map(new PseudoGrouping()) )
				   .mapPartitionsToPair(efun)  //execute parfor tasks, incl cleanup
		           .collect();                 //get output handles 
		
		//de-serialize results
		LocalVariableMap[] results = RemoteParForUtils.getResults(out, LOG);
		int numTasks = aTasks.value().intValue(); //get accumulator value
		int numIters = aIters.value().intValue(); //get accumulator value
		
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
	
	//determines if given input matrix requires grouping of partial partition slices
	private static boolean requiresGrouping(PDataPartitionFormat dpf, MatrixCharacteristics mc) {
		return (dpf == PDataPartitionFormat.ROW_WISE && mc.getNumColBlocks() > 1)
			|| (dpf == PDataPartitionFormat.COLUMN_WISE && mc.getNumRowBlocks() > 1);
	}
	
	//function to map data partition output to parfor input signature without grouping
	private static class PseudoGrouping implements Function<Tuple2<Long, Writable>, Tuple2<Long, Iterable<Writable>>>  {
		private static final long serialVersionUID = 2016614593596923995L;

		@Override
		public Tuple2<Long, Iterable<Writable>> call(Tuple2<Long, Writable> arg0) throws Exception {
			return new Tuple2<Long, Iterable<Writable>>(arg0._1(), Collections.singletonList(arg0._2()));
		}
	}
}
