/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.context.SparkExecutionContext;
import com.ibm.bi.dml.runtime.instructions.cp.Data;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		SparkExecutionContext sec = (SparkExecutionContext)ec;
		JavaSparkContext sc = sec.getSparkContext();
		
		//initialize accumulators for tasks/iterations
		Accumulator<Integer> aTasks = sc.accumulator(0);
		Accumulator<Integer> aIters = sc.accumulator(0);
		
		//run remote_spark parfor job 
		//(w/o lazy evaluation to fit existing parfor framework, e.g., result merge)
		RemoteParForSparkWorker func = new RemoteParForSparkWorker(pfid, program, cpCaching, aTasks, aIters);
		List<Tuple2<Long,String>> out = 
				sc.parallelize( tasks, numMappers )  //create rdd of parfor tasks
		          .flatMapToPair( func )             //execute parfor tasks 
		          .collect();                        //get output handles
		
		//de-serialize results
		LocalVariableMap[] results = getResults(out);
		int numTasks = aTasks.value(); //get accumulator value
		int numIters = aIters.value(); //get accumulator value
		
		//create output symbol table entries
		RemoteParForJobReturn ret = new RemoteParForJobReturn(true, numTasks, numIters, results);
		
		return ret;
	}
	
	/**
	 * 
	 * @param out
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	public static LocalVariableMap[] getResults( List<Tuple2<Long,String>> out ) 
		throws DMLRuntimeException
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<Long,LocalVariableMap>();

		int countAll = 0;
		for( Tuple2<Long,String> entry : out )
		{
			Long key = entry._1();
			String val = entry._2();
			if( !tmp.containsKey( key ) )
        		tmp.put(key, new LocalVariableMap ());	   
			Object[] dat = ProgramConverter.parseDataObject( val );
        	tmp.get(key).put((String)dat[0], (Data)dat[1]);
        	countAll++;
		}

		LOG.debug("Num remote worker results (before deduplication): "+countAll);
		LOG.debug("Num remote worker results: "+tmp.size());

		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);	
	}
}
