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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.ParForStatementBlock.ResultVar;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.lineage.Lineage;

/**
 * Super class for master/worker pattern implementations. Central place to
 * execute set or range tasks.
 * 
 */
public abstract class ParWorker
{
	protected static final Log LOG = LogFactory.getLog(ParWorker.class.getName());
	
	protected long                      _workerID    = -1;
	protected ArrayList<ProgramBlock>   _childBlocks = null;

	protected ExecutionContext          _ec          = null;
	protected ArrayList<ResultVar>      _resultVars  = null;
	protected long                      _numTasks    = -1;
	protected long                      _numIters    = -1;

	public ParWorker() {
		//implicit constructor (required if parameters not known on object creation, 
		//e.g., RemoteParWorkerMapper)
	}
	
	public ParWorker( long ID, ParForBody body ) {
		_workerID    = ID;
		if( body != null ) {
			_childBlocks = body.getChildBlocks();
			_ec = body.getEc();
			_resultVars = body.getResultVariables();
		}
		_numTasks = 0;
		_numIters = 0;
	}

	public ExecutionContext getExecutionContext() {
		return _ec;
	}
	
	public LocalVariableMap getVariables() {
		return _ec.getVariables();
	}
	
	/**
	 * Returns a summary statistic of executed tasks and hence should only be called 
	 * after execution.
	 * 
	 * @return number of executed tasks
	 */
	public long getExecutedTasks() {
		return _numTasks;
	}
	
	/**
	 * Returns a summary statistic of executed iterations and hence should only be called 
	 * after execution.
	 * 
	 * @return number of executed iterations
	 */
	public long getExecutedIterations() {
		return _numIters;
	}

	protected void pinResultVariables() {
		for( ResultVar var : _resultVars ) {
			Data dat = _ec.getVariable(var._name);
			if( dat instanceof MatrixObject )
				((MatrixObject)dat).enableCleanup(false);
		}
	}

	protected void executeTask( Task task ) {
		LOG.trace("EXECUTE PARFOR_WORKER ID="+_workerID+" for task "+task.toCompactString());
		
		switch( task.getType() ) {
			case SET:
				executeSetTask( task );
				break;
			case RANGE:
				executeRangeTask( task );
				break;
		}
	}	

	private void executeSetTask( Task task ) {
		//foreach iteration in task, execute iteration body
		String lVarName = task.getVarName();
		for( IntObject indexVal : task.getIterations() ) {
			//set index values
			_ec.setVariable(lVarName, indexVal);
			if (DMLScript.LINEAGE) {
				Lineage li = _ec.getLineage();
				li.set(lVarName, li.getOrCreate(new CPOperand(indexVal)));
			}
			
			// for each program block
			for (ProgramBlock pb : _childBlocks)
				pb.execute(_ec);
					
			_numIters++;
		}

		_numTasks++;
	}

	private void executeRangeTask( Task task ) {
		List<IntObject> tmp = task.getIterations();
		String lVarName = task.getVarName();
		long lFrom      = tmp.get(0).getLongValue();
		long lTo        = tmp.get(1).getLongValue();
		long lIncr      = tmp.get(2).getLongValue();
		
		for( long i=lFrom; i<=lTo; i+=lIncr ) {
			//set index values
			IntObject indexVal = new IntObject(i);
			_ec.setVariable(lVarName, indexVal);
			if (DMLScript.LINEAGE) {
				Lineage li = _ec.getLineage();
				li.set(lVarName, li.getOrCreate(new CPOperand(indexVal)));
			}
			
			// for each program block
			for (ProgramBlock pb : _childBlocks)
				pb.execute(_ec);
			
			_numIters++;
		}

		_numTasks++;
	}
}
