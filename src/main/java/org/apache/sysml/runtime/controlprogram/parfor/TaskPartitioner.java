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

import java.util.List;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.IntObject;

/**
 * This is the base class for all task partitioner. For this purpose it stores relevant information such as
 * the loop specification (FROM, TO, INCR), the index variable and the task size. Furthermore, it declares two
 * prototypes: (1) full task creation, (2) streaming task creation.
 * 
 * Known implementation classes: TaskPartitionerFixedsize, TaskPartitionerFactoring
 * 
 */
public abstract class TaskPartitioner 
{	
	protected long           _taskSize     = -1;	
	protected String  		 _iterVarName  = null;
	protected IntObject      _fromVal      = null;
	protected IntObject      _toVal        = null;
	protected IntObject      _incrVal      = null;
	protected long           _numIter      = -1;
	
	protected TaskPartitioner( long taskSize, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		_taskSize    = taskSize;
		_iterVarName = iterVarName;
		_fromVal     = fromVal;
		_toVal       = toVal;
		_incrVal     = incrVal;
		
		//normalize predicate if necessary
		normalizePredicate();
		
		//compute number of iterations
		_numIter     = (long)Math.ceil(((double)(_toVal.getLongValue()-_fromVal.getLongValue()+1 )) / _incrVal.getLongValue()); 
	}
	
	/**
	 * Creates and returns set of all tasks for given problem at once.
	 * 
	 * @return
	 */
	public abstract List<Task> createTasks()
		throws DMLRuntimeException;
	
	/**
	 * Creates set of all tasks for given problem, but streams them directly
	 * into task queue. This allows for more tasks than fitting in main memory.
	 * 
	 * @return
	 */
	public abstract long createTasks( LocalTaskQueue<Task> queue )
		throws DMLRuntimeException;

	
	/**
	 * 
	 * @return
	 */
	public long getNumIterations() {
		return _numIter;
	}
	
	/**
	 * Normalizes the (from, to, incr) predicate to a predicate w/
	 * positive increment.
	 */
	private void normalizePredicate() {
		//check for positive increment
		if( _incrVal.getLongValue() >= 0 )
			return;
		
		long lfrom = _fromVal.getLongValue();
		long lto = _toVal.getLongValue();
		long lincr = _incrVal.getLongValue();
		
		_fromVal = new IntObject(lfrom - ((lfrom - lto)/lincr * lincr));
		_toVal   = new IntObject(lfrom);
		_incrVal = new IntObject(-1 * lincr);
	}
}
