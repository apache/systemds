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


/**
 * Lop to perform transpose-identity operation (t(X)%*%X or X%*%t(X)),
 * used to represent CP and MR instruction but in case of MR there is
 * an additional Aggregate at the reducers.
 */
public class MMTSJ extends Lop 
{
	public enum MMTSJType {
		NONE,
		LEFT,
		RIGHT;
		
		public boolean isLeft(){
			return (this == LEFT);
		}
		public boolean isRight(){
			return (this == RIGHT);
		}
	}
	
	private MMTSJType _type = null;
	private boolean _multiPass = false;
	private int _numThreads = 1;

	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type) {
		this(input1, dt, vt, et, type, false, -1);
	}
	
	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type, boolean multiPass) {
		this(input1, dt, vt, et, type, multiPass, -1);
	}
	
	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type, boolean multiPass, int k) {
		super(Lop.Type.MMTSJ, dt, vt);		
		addInput(input1);
		input1.addOutput(this);
		_type = type;
		_multiPass = multiPass;
		_numThreads = k;
		
		if( multiPass && et != ExecType.SPARK )
			throw new RuntimeException("Multipass tsmm only supported for exec type SPARK.");
		 
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return "Operation = MMTSJ";
	}

	@Override
	public String getInstructions(String input_index1, String output_index)
	{	
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _multiPass ? "tsmm2" : Opcodes.TSMM.toString() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input_index1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output_index));
		sb.append( OPERAND_DELIMITOR );
		sb.append( _type );
		
		//append degree of parallelism for matrix multiplications
		if( getExecType()==ExecType.CP || getExecType()==ExecType.FED ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
			if ( getExecType()==ExecType.FED ){
				sb.append( OPERAND_DELIMITOR );
				sb.append( _fedOutput.name() );
			}
		}
		
		return sb.toString();
	}
}
