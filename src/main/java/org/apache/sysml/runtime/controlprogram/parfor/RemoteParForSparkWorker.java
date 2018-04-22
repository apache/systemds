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
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.codegen.CodegenUtils;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.caching.FrameObject;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.runtime.util.UtilFunctions;

import scala.Tuple2;

public class RemoteParForSparkWorker extends ParWorker implements PairFlatMapFunction<Task, Long, String> 
{
	private static final long serialVersionUID = -3254950138084272296L;
	
	private static final CachedReuseVariables reuseVars = new CachedReuseVariables();
	
	private final long _jobid;
	private final String _prog;
	private final HashMap<String, byte[]> _clsMap;
	private boolean _initialized = false;
	private boolean _caching = true;
	
	private final LongAccumulator _aTasks;
	private final LongAccumulator _aIters;

	private final Map<String, Broadcast> _brInputs;
	
	public RemoteParForSparkWorker(long jobid, String program, HashMap<String, byte[]> clsMap, boolean cpCaching, LongAccumulator atasks, LongAccumulator aiters, Map<String, Broadcast> brInputs) {
		_jobid = jobid;
		_prog = program;
		_clsMap = clsMap;
		_initialized = false;
		_caching = cpCaching;
		//setup spark accumulators
		_aTasks = atasks;
		_aIters = aiters;
		_brInputs = brInputs;
	}
	
	@Override 
	public Iterator<Tuple2<Long, String>> call(Task arg0)
		throws Exception 
	{
		//lazy parworker initialization
		if( !_initialized )
			configureWorker(TaskContext.get().taskAttemptId());
		
		//execute a single task
		long numIter = getExecutedIterations();
		super.executeTask( arg0 );
		
		//maintain accumulators
		_aTasks.add( 1 );
		_aIters.add( (int)(getExecutedIterations()-numIter) );
		
		//write output if required (matrix indexed write) 
		//note: this copy is necessary for environments without spark libraries
		ArrayList<Tuple2<Long,String>> ret = new ArrayList<>();
		ArrayList<String> tmp = RemoteParForUtils.exportResultVariables( _workerID, _ec.getVariables(), _resultVars );
		for( String val : tmp )
			ret.add(new Tuple2<>(_workerID, val));
		
		return ret.iterator();
	}
	
	private void configureWorker(long taskID) 
		throws IOException
	{
		_workerID = taskID;
		
		//initialize codegen class cache (before program parsing)
		for( Entry<String, byte[]> e : _clsMap.entrySet() )
			CodegenUtils.getClassSync(e.getKey(), e.getValue());
	
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();
		_resultVars  = body.getResultVariables();
		_numTasks    = 0;
		_numIters    = 0;

		//fetch the broadcast variables
		if (OptimizerUtils.ALLOW_BROADCAST_INPUTS_PAR_FOR && !reuseVars.containsVariables(_jobid)) {
			loadBroadcastVariables(_ec.getVariables(), _brInputs);
		}

		//reuse shared inputs (to read shared inputs once per process instead of once per core; 
		//we reuse everything except result variables and partitioned input matrices)
		_ec.pinVariables(_ec.getVarList()); //avoid cleanup of shared inputs
		Collection<String> blacklist = UtilFunctions.asSet(_resultVars.stream()
			.map(v -> v._name).collect(Collectors.toList()), _ec.getVarListPartitioned());
		reuseVars.reuseVariables(_jobid, _ec.getVariables(), blacklist);
		
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

	private void loadBroadcastVariables(LocalVariableMap variables, Map<String, Broadcast> _brInputs) {
		for (String key : variables.keySet()) {
			if (!_brInputs.containsKey(key)) {
				continue;
			}
			Data d = variables.get(key);
			if (d instanceof MatrixObject) {
				MatrixObject mo = (MatrixObject) d;
				MatrixBlock mb = (MatrixBlock) _brInputs.get(key).getValue();
				mo.acquireModify(mb);
				mo.setEmptyStatus();// set status from modified to empty
				mo.updateMatrixCharacteristics(new MatrixCharacteristics(mb.getNumRows(), mb.getNumColumns(), 1, 1));
			} else if (d instanceof FrameObject) {
				FrameObject fo = (FrameObject) d;
				FrameBlock fb = (FrameBlock) _brInputs.get(key).getValue();
				fo.acquireModify(fb);
				fo.setEmptyStatus(); // set status from modified to empty
				fo.updateMatrixCharacteristics(new MatrixCharacteristics(fb.getNumRows(), fb.getNumColumns(), 1, 1));
			}
		}
	}

	public static void cleanupCachedVariables(long pfid) {
		reuseVars.clearVariables(pfid);
	}
}
