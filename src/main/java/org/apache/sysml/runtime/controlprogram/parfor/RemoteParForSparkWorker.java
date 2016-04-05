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

import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.Accumulator;
import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.util.LocalFileUtils;

import scala.Tuple2;

/**
 * 
 * 
 */
public class RemoteParForSparkWorker extends ParWorker implements PairFlatMapFunction<Task, Long, String> 
{
	
	private static final long serialVersionUID = -3254950138084272296L;

	private boolean _initialized = false;
	private String  _prog = null;
	private boolean _caching = true;
	
	private Accumulator<Integer> _aTasks = null;
	private Accumulator<Integer> _aIters = null;
	
	public RemoteParForSparkWorker(String program, boolean cpCaching, Accumulator<Integer> atasks, Accumulator<Integer> aiters) 
		throws DMLRuntimeException
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
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void configureWorker( long ID ) 
		throws DMLRuntimeException, IOException
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
		if( !CacheableData.isCachingActive() ) {
			String uuid = IDHandler.createDistributedUniqueID();
			LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
			CacheableData.initCaching( uuid ); //incl activation, cache dir creation (each map task gets its own dir for simplified cleanup)
		}		
		if( !CacheableData.cacheEvictionLocalFilePrefix.contains("_") ){ //account for local mode
			CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
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
