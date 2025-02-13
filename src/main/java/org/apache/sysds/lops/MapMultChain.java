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
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;


public class MapMultChain extends Lop 
{
	public static final String OPCODE = Opcodes.MAPMMCHAIN.toString();
	public static final String OPCODE_CP = Opcodes.MMCHAIN.toString();

	public enum ChainType {
		XtXv,  //(t(X) %*% (X %*% v))
		XtwXv, //(t(X) %*% (w * (X %*% v)))
		XtXvy, //(t(X) %*% ((X %*% v) - y))
		NONE;
		public boolean isWeighted() {
			return this == XtwXv || this == ChainType.XtXvy;
		}
	}
	
	private ChainType _chainType = null;
	private int _numThreads = 1;
	
	/**
	 * Constructor to setup a map mult chain without weights
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public MapMultChain(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et)  {
		super(Lop.Type.MapMultChain, dt, vt);
		addInput(input1); //X
		addInput(input2); //v
		input1.addOutput(this); 
		input2.addOutput(this); 
		
		//setup mapmult parameters
		_chainType = ChainType.XtXv;
		setupLopProperties(et);
	}
	
	/**
	 * Constructor to setup a map mult chain with weights
	 * 
	 * @param input1 low-level operator 1
	 * @param input2 low-level operator 2
	 * @param input3 low-level operator 3
	 * @param chain chain type
	 * @param dt data type
	 * @param vt value type
	 * @param et execution type
	 */
	public MapMultChain(Lop input1, Lop input2, Lop input3, ChainType chain, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.MapMultChain, dt, vt);
		addInput(input1); //X
		addInput(input2); //w
		addInput(input3); //v
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		//setup mapmult parameters
		_chainType = chain;
		setupLopProperties(et);
	}
	
	public void setNumThreads(int k) {
		_numThreads = k;
	}
	
	@Override
	public String toString() {
		return "Operation = MapMMChain";
	}

	@Override
	public Lop getBroadcastInput() {
		if (getExecType() != ExecType.SPARK)
			return null;
		return getInputs().get(1);
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		return getInstructions(input1, input2, null, output);
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output)
	{
		//Spark instruction XtwXv
		StringBuilder sb = InstructionUtils.getStringBuilder();
		
		sb.append(getExecType());
		sb.append(Lop.OPERAND_DELIMITOR);
		
		if( getExecType()==ExecType.CP )
			sb.append(OPCODE_CP);
		else
			sb.append(OPCODE);
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append( getInputs().get(1).prepInputOperand(input2));
		
		if( input3 != null ) {
			sb.append(Lop.OPERAND_DELIMITOR);
			sb.append( getInputs().get(2).prepInputOperand(input3));
		}
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(prepOutputOperand(output));
		
		sb.append(Lop.OPERAND_DELIMITOR);
		sb.append(_chainType);
		
		//append degree of parallelism for matrix multiplications
		if( getExecType()==ExecType.CP ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		
		return sb.toString();
	}
}
