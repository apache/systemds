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
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class CumulativeOffsetBinary extends Lop 
{
	private AggOp _op;
	private double _initValue = 0;
	private boolean _broadcast = false;
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, AggOp op, ExecType et) 
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
	
		init(data, offsets, dt, vt, et);
	}
	
	public CumulativeOffsetBinary(Lop data, Lop offsets, DataType dt, ValueType vt, double init, boolean broadcast, AggOp op, ExecType et)
	{
		super(Lop.Type.CumulativeOffsetBinary, dt, vt);
		checkSupportedOperations(op);
		_op = op;
		
		//in case of Spark, CumulativeOffset includes CumulativeSplit and hence needs the init value
		_initValue = init;
		_broadcast = broadcast;
		
		init(data, offsets, dt, vt, et);
	}
	
	private void init(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		addInput(input1);
		addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return "CumulativeOffsetBinary";
	}

	private static void checkSupportedOperations(AggOp op) {
		//sanity check for supported aggregates
		if( !( op == AggOp.SUM || op == AggOp.PROD
			|| op == AggOp.SUM_PROD
			|| op == AggOp.MIN || op == AggOp.MAX) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
	}
	
	private String getOpcode() {
		switch( _op ) {
			case SUM:      return Opcodes.BCUMOFFKP.toString();
			case PROD:     return Opcodes.BCUMOFFM.toString();
			case SUM_PROD: return Opcodes.BCUMOFFPM.toString();
			case MIN:      return Opcodes.BCUMOFFMIN.toString();
			case MAX:      return Opcodes.BCUMOFFMAX.toString();
			default:       return null;
		}
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output)
	{
		String inst = InstructionUtils.concatOperands(
			getExecType().name(), getOpcode(),
			getInputs().get(0).prepInputOperand(input1),
			getInputs().get(1).prepInputOperand(input2),
			prepOutputOperand(output) );
		
		if( getExecType() == ExecType.SPARK ) {
			inst = InstructionUtils.concatOperands(inst,
				String.valueOf(_initValue), String.valueOf(_broadcast) );
		}
		
		return inst;
	}
}
