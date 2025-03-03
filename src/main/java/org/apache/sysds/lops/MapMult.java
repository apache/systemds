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

package org.apache.sysds.lops;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.hops.AggBinaryOp.SparkAggType;
 
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


public class MapMult extends Lop 
{
	public static final String OPCODE = Opcodes.MAPMM.toString();
	
	public enum CacheType {
		RIGHT,
		RIGHT_PART,
		LEFT,
		LEFT_PART;
		
		public boolean isRight() {
			return (this == RIGHT || this == RIGHT_PART);
		}
		
		public CacheType getFlipped() {
			switch( this ) {
				case RIGHT: return LEFT;
				case RIGHT_PART: return LEFT_PART;
				case LEFT: return RIGHT;
				case LEFT_PART: return RIGHT_PART;
				default: return null;
			}
		}
	}
	
	private CacheType _cacheType = null;
	private boolean _outputEmptyBlocks = true;
	
	//optional attribute for spark exec type
	private SparkAggType _aggtype = SparkAggType.MULTI_BLOCK;

	/**
	 * Constructor to setup a partial Matrix-Vector Multiplication for Spark
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param dt data type
	 * @param vt value type
	 * @param rightCache true if right cache, false if left cache
	 * @param partitioned true if partitioned, false if not partitioned
	 * @param emptyBlocks true if output empty blocks
	 * @param aggtype spark aggregation type
	 */
	public MapMult(Lop input1, Lop input2, DataType dt, ValueType vt, boolean rightCache, boolean partitioned, boolean emptyBlocks, SparkAggType aggtype) {
		super(Lop.Type.MapMult, dt, vt);
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		//setup mapmult parameters
		if( rightCache )
			_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
		else
			_cacheType = partitioned ? CacheType.LEFT_PART : CacheType.LEFT;
		_outputEmptyBlocks = emptyBlocks;
		_aggtype = aggtype;
		
		lps.setProperties( inputs, ExecType.SPARK);
	}

	@Override
	public SparkAggType getAggType() {
		return _aggtype;
	}
	
	@Override
	public Lop getBroadcastInput() {
		if (getExecType() != ExecType.SPARK)
			return null;
		
		return _cacheType.isRight() ? getInputs().get(1) : getInputs().get(0);
		//Note: rdd and broadcast inputs can flip during runtime
	}

	@Override
	public String toString() {
		return "Operation = MapMM";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		String ret = InstructionUtils.concatOperands(
			getExecType().name(), OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output),
			_cacheType.name(),
			String.valueOf(_outputEmptyBlocks));
		
		if( getExecType() == ExecType.SPARK )
			ret = InstructionUtils.concatOperands(ret, _aggtype.name());
		
		return ret;
	}
}
