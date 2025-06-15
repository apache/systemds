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

import java.util.HashMap;


import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecType;

import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;


/**
 * Lop to perform mr map-side grouped aggregates 
 * (restriction: sum, w/o weights, ngroups), groups broadcasted
 * 
 */
public class GroupedAggregateM extends Lop 
{	
	public static final String OPCODE = Opcodes.MAPGROUPEDAGG.toString();

	public enum CacheType {
		RIGHT,
		RIGHT_PART,
	}
	
	private HashMap<String, Lop> _inputParams;
	private CacheType _cacheType = null;
		
	public GroupedAggregateM(HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, boolean partitioned, ExecType et) {
		super(Lop.Type.GroupedAggM, dt, vt);
		init(inputParameterLops, dt, vt, et);
		_inputParams = inputParameterLops;
		_cacheType = partitioned ? CacheType.RIGHT_PART : CacheType.RIGHT;
	}

	private void init(HashMap<String, Lop> inputParameterLops, 
			DataType dt, ValueType vt, ExecType et) 
	{
		addInput(inputParameterLops.get(Statement.GAGG_TARGET));
		inputParameterLops.get(Statement.GAGG_TARGET).addOutput(this);
		addInput(inputParameterLops.get(Statement.GAGG_GROUPS));
		inputParameterLops.get(Statement.GAGG_GROUPS).addOutput(this);
		
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return "Operation = MapGroupedAggregate";
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			OPCODE,
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output),
			_inputParams.get(Statement.GAGG_NUM_GROUPS).prepScalarInputOperand(getExecType()),
			_cacheType.name());
	}
}