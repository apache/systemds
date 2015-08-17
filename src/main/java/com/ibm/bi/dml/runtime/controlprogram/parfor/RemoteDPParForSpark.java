/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.utils.Statistics;

/**
 * TODO robustness on failures (cleanup files)
 * TODO heavy hitter maintenance
 * TODO data partitioning with binarycell
 *
 */
public class RemoteDPParForSpark
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		int rlen = (int)input.getNumRows();
		int clen = (int)input.getNumColumns();
		int brlen = (int)input.getNumRowsPerBlock();
		int bclen = (int)input.getNumColumnsPerBlock();
		InputInfo ii = InputInfo.BinaryBlockInputInfo;
				
		//initialize accumulators for tasks/iterations
		Accumulator<Integer> aTasks = sc.accumulator(0);
		Accumulator<Integer> aIters = sc.accumulator(0);
		
		JavaPairRDD<MatrixIndexes,MatrixBlock> in = sec.getBinaryBlockRDDHandleForVariable(matrixvar);
		DataPartitionerRemoteSparkMapper dpfun = new DataPartitionerRemoteSparkMapper(rlen, clen, brlen, bclen, ii, oi, dpf);
		RemoteDPParForSparkWorker efun = new RemoteDPParForSparkWorker(program, matrixvar, itervar, 
				          enableCPCaching, rlen, clen, brlen, bclen, tSparseCol, dpf, oi, aTasks, aIters);
		List<Tuple2<Long,String>> out = 
				in.flatMapToPair(dpfun)    //partition the input blocks
		          .groupByKey(numReducers) //group partition blocks 		          
		          .flatMapToPair( efun )   //execute parfor tasks 
		          .collect();              //get output handles
		
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
