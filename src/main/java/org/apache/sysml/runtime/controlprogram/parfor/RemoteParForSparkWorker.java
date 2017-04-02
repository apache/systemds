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
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.util.LocalFileUtils;

import scala.Tuple2;

public class RemoteParForSparkWorker extends ParWorker implements PairFlatMapFunction<Task, Long, String> 
{
	private static final long serialVersionUID = -3254950138084272296L;

	private final String  _prog;
	private final HashMap<String, byte[]> _clsMap;
	private boolean _initialized = false;
	private boolean _caching = true;
	
	private final LongAccumulator _aTasks;
	private final LongAccumulator _aIters;
	
	public RemoteParForSparkWorker(String program, HashMap<String, byte[]> clsMap, boolean cpCaching, LongAccumulator atasks, LongAccumulator aiters) 
		throws DMLRuntimeException
	{
		_prog = program;
		_clsMap = clsMap;
		_initialized = false;
		_caching = cpCaching;
		
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
	}
	
	@Override 
	public Iterator<Tuple2<Long, String>> call(Task arg0)
		throws Exception 
	{
		//lazy parworker initialization
		if( !_initialized )
			configureWorker( TaskContext.get().taskAttemptId() );
		
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
			
		return ret.iterator();
	}

	private void configureWorker( long ID ) 
		throws DMLRuntimeException, IOException
	{
		_workerID = ID;
		
		//initialize codegen class cache (before program parsing)
		synchronized( CodegenUtils.class ) {
			for( Entry<String, byte[]> e : _clsMap.entrySet() )
				CodegenUtils.getClass(e.getKey(), e.getValue());
		}
		
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();				
		_resultVars  = body.getResultVarNames();
		_numTasks    = 0;
		_numIters    = 0;

		//init and register-cleanup of buffer pool (in parfor spark, multiple tasks might 
		//share the process-local, i.e., per executor, buffer pool; hence we synchronize 
		//the initialization and immediately register the created directory for cleanup
		//on process exit, i.e., executor exit, including any files created in the future.
		synchronized( CacheableData.class ) {
			if( !CacheableData.isCachingActive() && !InfrastructureAnalyzer.isLocalMode() ) { 
				//create id, executor working dir, and cache dir
				String uuid = IDHandler.createDistributedUniqueID();
				LocalFileUtils.createWorkingDirectoryWithUUID( uuid );
				CacheableData.initCaching( uuid ); //incl activation and cache dir creation
				CacheableData.cacheEvictionLocalFilePrefix = 
						CacheableData.cacheEvictionLocalFilePrefix +"_" + _workerID; 
				//register entire working dir for delete on shutdown
				RemoteParForUtils.cleanupWorkingDirectoriesOnShutdown();
			}	
		}
		
		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required and not in CP process)
		if( !_caching && !InfrastructureAnalyzer.isLocalMode() )
			CacheableData.disableCaching();
		
		//mark as initialized
		_initialized = true;
	}
}
