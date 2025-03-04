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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;

import java.io.Serializable;

public abstract class ResultMerge<T extends CacheableData<?>> implements Serializable
{
	//note: this class needs to be serializable to ensure that all attributes of
	//ResultMergeRemoteSparkWCompare are included in the task closure
	private static final long serialVersionUID = -6756689640511059030L;
	
	protected static final Log LOG = LogFactory.getLog(ResultMerge.class.getName());
	protected static final String NAME_SUFFIX = "_rm";
	protected static final BinaryOperator PLUS = InstructionUtils.parseBinaryOperator("+");
	protected static final BinaryOperator MINUS = InstructionUtils.parseBinaryOperator("-");
	
	//inputs to result merge
	protected T       _output      = null;
	protected T[]     _inputs      = null; 
	protected String  _outputFName = null;
	protected boolean _isAccum     = false;
	
	protected ResultMerge( ) {
		//do nothing
	}
	
	public ResultMerge( T out, T[] in, String outputFilename, boolean accum ) {
		_output = out;
		_inputs = in;
		_outputFName = outputFilename;
		_isAccum = accum;
	}
	
	/**
	 * Merge all given input matrices sequentially into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of one input matrix at a time.
	 * 
	 * @return output (merged) matrix
	 */
	public abstract T executeSerialMerge();
	
	/**
	 * Merge all given input matrices in parallel into the given output matrix.
	 * The required space in-memory is the size of the output matrix plus the size
	 * of all input matrices.
	 * 
	 * @param par degree of parallelism
	 * @return output (merged) matrix
	 */
	public abstract T executeParallelMerge(int par);
	
}
