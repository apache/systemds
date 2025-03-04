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

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class WeightedUnaryMM extends Lop 
{
	public static final String OPCODE = "mapwumm";
	public static final String OPCODE_CP = Opcodes.WUMM.toString();

	public enum WUMMType {
		MULT,
		DIV,
	}
	
	private WUMMType _wummType = null;
	private OpOp1 _uop = null;
	private int _numThreads = 1;
	
	public WeightedUnaryMM(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, WUMMType wt, OpOp1 op, ExecType et) {
		super(Lop.Type.WeightedUMM, dt, vt);
		addInput(input1); //X
		addInput(input2); //U
		addInput(input3); //V
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		//setup mapmult parameters
		_wummType = wt;
		_uop = op;
		setupLopProperties(et);
	}
	
	@Override
	public String toString() {
		return "Operation = WeightedUMM";
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		
		sb.append(getExecType());
		
		sb.append(Lop.OPERAND_DELIMITOR);
		if( getExecType() == ExecType.CP )
			sb.append(OPCODE_CP);
		else
			sb.append(OPCODE);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_uop.toString());
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input2));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(2).prepInputOperand(input3));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( prepOutputOperand(output));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_wummType);
		
		//append degree of parallelism
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		
		return sb.toString();
	}
	
	public void setNumThreads(int k) {
		_numThreads = k;
	}
}
