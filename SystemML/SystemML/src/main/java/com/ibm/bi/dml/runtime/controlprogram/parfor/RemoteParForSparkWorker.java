/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.Accumulator;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheableData;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.util.LocalFileUtils;

import scala.Tuple2;

/**
 * 
 * 
 */
public class RemoteParForSparkWorker extends ParWorker implements PairFlatMapFunction<Task, Long, String> 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long serialVersionUID = -3254950138084272296L;

	private boolean _initialized = false;
	private String  _prog = null;
	private boolean _caching = true;
	
	private Accumulator<Integer> _aTasks = null;
	private Accumulator<Integer> _aIters = null;
	
	public RemoteParForSparkWorker(String program, boolean cpCaching, Accumulator<Integer> atasks, Accumulator<Integer> aiters) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//keep inputs (unfortunately, spark does not expose task ids and it would be implementation-dependent
		//when this constructor is actually called; hence, we do lazy initialization on task execution)
		_initialized = false;
		_prog = program;
		_caching = cpCaching;
		
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
	}
	
	@Override 
	public Iterable<Tuple2<Long, String>> call(Task arg0)
		throws Exception 
	{
		//lazy parworker initialization
		if( !_initialized )
			configureWorker( TaskContext.get().taskAttemptId() ); //requires Spark 1.3
		
		//execute a single task
		long numIter = getExecutedIterations();
		super.executeTask( arg0 );
		
		//maintain accumulators
		_aTasks.add( 1 );
		_aIters.add( (int)(getExecutedIterations()-numIter) );
		
		//write output if required (matrix indexed write) 
		//note: this copy is necessary for environments without spark libraries
		ArrayList<Tuple2<Long,String>> ret = new ArrayList<Tuple2<Long,String>>();
		ArrayList<String> tmp = RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars );
		for( String val : tmp )
			ret.add(new Tuple2<Long,String>(_workerID, val));
			
		return ret;
	}
	
	/**
	 * 
	 * @param ID
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void configureWorker( long ID ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, IOException
	{
		_workerID = ID;
		
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();				
		_resultVars  = body.getResultVarNames();
		_numTasks    = 0;
		_numIters    = 0;

		//init local cache manager 
		if( !CacheableData.isCachingActive() ) 
		{
			String uuid = IDHandler.createDistributedUniqueID();
			LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
			CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
		}		
		if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ) //account for local mode
		{
			CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
			CacheableData.cacheEvictionHDFSFilePrefix = CacheableData.cacheEvictionHDFSFilePrefix +"_" + _workerID;
		}
		
		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required)
		if( !_caching )
			CacheableData.disableCaching();
		
		//make as lazily intialized
		_initialized = true;
	}
}
