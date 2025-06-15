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
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

public class CumulativePartialAggregate extends Lop 
{
	private AggOp _op;
	
	public CumulativePartialAggregate(Lop input, DataType dt, ValueType vt, AggOp op, ExecType et) {
		super(Lop.Type.CumulativePartialAggregate, dt, vt);
		
		//sanity check for supported aggregates
		if( !( op == AggOp.SUM || op == AggOp.PROD 
			|| op == AggOp.SUM_PROD
			|| op == AggOp.MIN || op == AggOp.MAX) )
		{
			throw new LopsException("Unsupported aggregate operation type: "+op);
		}
		_op = op;
		init(input, dt, vt, et);
	}
	
	private void init(Lop input, DataType dt, ValueType vt, ExecType et) {
		addInput(input);
		input.addOutput(this);
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {
		return "CumulativePartialAggregate";
	}
	
	private String getOpcode() {
		switch( _op ) {
			case SUM:      return Opcodes.UCUMACKP.toString();
			case PROD:     return Opcodes.UCUMACM.toString();
			case SUM_PROD: return Opcodes.UCUMACPM.toString();
			case MIN:      return Opcodes.UCUMACMIN.toString();
			case MAX:      return Opcodes.UCUMACMAX.toString();
			default:       return null;
		}
	}
	
	@Override
	public String getInstructions(String input, String output) {
		return InstructionUtils.concatOperands(
			getExecType().name(),
			getOpcode(),
			getInputs().get(0).prepInputOperand(input),
			prepOutputOperand(output));
	}
}
