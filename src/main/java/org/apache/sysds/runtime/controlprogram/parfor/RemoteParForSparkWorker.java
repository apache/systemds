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

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.spark.TaskContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.util.LongAccumulator;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.dml.DmlSyntacticValidator;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.util.CollectionUtils;
import org.apache.sysds.runtime.util.ProgramConverter;

import scala.Tuple2;

public class RemoteParForSparkWorker extends ParWorker implements PairFlatMapFunction<Task, Long, String> 
{
	private static final long serialVersionUID = -3254950138084272296L;
	
	private static final CachedReuseVariables reuseVars = new CachedReuseVariables();
	
	private final long _jobid;
	private final String _prog;
	private final boolean _isLocal;
	private final HashMap<String, byte[]> _clsMap;
	private boolean _initialized = false;
	private boolean _caching = true;
	private final boolean _cleanCache;
	private final Map<String,String> _lineage;
	
	private final LongAccumulator _aTasks;
	private final LongAccumulator _aIters;

	private final Map<String, Broadcast<CacheBlock<?>>> _brInputs;
	
	public RemoteParForSparkWorker(long jobid, String program, boolean isLocal,
		HashMap<String, byte[]> clsMap, boolean cpCaching, LongAccumulator atasks, LongAccumulator aiters,
		Map<String, Broadcast<CacheBlock<?>>> brInputs, boolean cleanCache, Map<String,String> lineage) 
	{
		_jobid = jobid;
		_prog = program;
		_isLocal = isLocal;
		_clsMap = clsMap;
		_initialized = false;
		_caching = cpCaching;
		_aTasks = atasks;
		_aIters = aiters;
		_brInputs = brInputs;
		_cleanCache = cleanCache;
		_lineage = lineage;
	}
	
	@Override 
	public Iterator<Tuple2<Long, String>> call(Task arg0)
		throws Exception 
	{
		//lazy parworker initialization
		if( !_initialized )
			configureWorker(TaskContext.get().taskAttemptId());
		
		//keep input var names
		Set<String> inVars = new HashSet<>(_ec.getVariables().keySet());
		
		//execute a single task
		long numIter = getExecutedIterations();
		super.executeTask( arg0 );
		
		//maintain accumulators
		_aTasks.add( 1 );
		_aIters.add( (int)(getExecutedIterations()-numIter) );
		
		//cleanup remaining intermediate variables from buffer pool
		_ec.getVariables().keySet().stream().filter(v -> !inVars.contains(v))
			.map(v -> _ec.getVariable(v)).filter(d -> d instanceof CacheableData)
			.forEach(c -> ((CacheableData<?>)c).freeEvictedBlob());
		
		//write output lineage of required
		if( DMLScript.LINEAGE )
			RemoteParForUtils.exportLineageItems(_workerID, 
				_ec.getVariables(), _resultVars, _ec.getLineage());
		
		//write output if required (matrix indexed write), incl cleanup pinned vars
		//note: this copy is necessary for environments without spark libraries
		return RemoteParForUtils
			.exportResultVariables(_workerID, _ec.getVariables(), _resultVars)
			.stream().map(s -> new Tuple2<>(_workerID, s)).iterator();
	}
	
	private void configureWorker(long taskID) 
		throws IOException
	{
		_workerID = taskID;
		
		//initialize codegen class cache (before program parsing)
		for( Entry<String, byte[]> e : _clsMap.entrySet() )
			CodegenUtils.getClassSync(e.getKey(), e.getValue());
	
		//parse and setup parfor body program
		ParForBody body = ProgramConverter.parseParForBody(_prog, (int)_workerID, true);
		_childBlocks = body.getChildBlocks();
		_ec          = body.getEc();
		_resultVars  = body.getResultVariables();
		_numTasks    = 0;
		_numIters    = 0;

		//reuse shared inputs (to read shared inputs once per process instead of once per core; 
		//we reuse everything except result variables and partitioned input matrices)
		Collection<String> excludeList = CollectionUtils.asSet(_resultVars.stream()
			.map(v -> v._name).collect(Collectors.toList()), _ec.getVarListPartitioned());
		reuseVars.reuseVariables(_jobid, _ec.getVariables(), excludeList, _brInputs, _cleanCache);
		
		//setup the buffer pool
		RemoteParForUtils.setupBufferPool(_workerID, _isLocal);

		//ensure that resultvar files are not removed
		super.pinResultVariables();
		
		//enable/disable caching (if required and not in CP process)
		if( !_caching && !_isLocal )
			CacheableData.disableCaching();
		
		//ensure local mode for eval function loading on demand,
		//and reset thread-local memory of loaded functions (new dictionary)
		if( !_isLocal )
			DMLScript.setGlobalExecMode(ExecMode.SINGLE_NODE);
		DmlSyntacticValidator.init();

		//enable and setup lineage
		if( _lineage != null ) {
			DMLScript.LINEAGE = true;
			_ec.setLineage(Lineage.deserialize(_lineage));
		}
		
		//mark as initialized
		_initialized = true;
	}

	public static void cleanupCachedVariables(long pfid) {
		reuseVars.clearVariables(pfid);
	}
}
