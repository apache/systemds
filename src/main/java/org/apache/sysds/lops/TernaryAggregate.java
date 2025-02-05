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
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;

public class TernaryAggregate extends Lop 
{
	public static final String OPCODE_RC = Opcodes.TAKPM.toString();
	public static final String OPCODE_C = Opcodes.TACKPM.toString();
	
	//NOTE: currently only used for ta+*
	//private AggOp _aggOp = null;
	//private Binary.OperationTypes _binOp = null;
	private Direction _direction;
	
	//optional attribute for cp
	private int _numThreads = -1;

	public TernaryAggregate(Lop input1, Lop input2, Lop input3, AggOp aggOp, OpOp2 binOp, Direction direction, DataType dt, ValueType vt, ExecType et, int k ) 
	{
		super(Lop.Type.TernaryAggregate, dt, vt);
		
		//_aggOp = aggOp;
		//_binOp = binOp;
		
		addInput(input1);
		addInput(input2);
		addInput(input3);
		input1.addOutput(this);
		input2.addOutput(this);
		input3.addOutput(this);
		
		_direction = direction;
		_numThreads = k;
		
		lps.setProperties( inputs, et);
	}
	
	@Override
	public String toString() {
		return "Operation: "+getOpCode();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String output)
	{
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpCode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepInputOperand(input3));
		sb.append( OPERAND_DELIMITOR );
		sb.append( prepOutputOperand(output));
		
		if( getExecType() == ExecType.CP || getExecType() == ExecType.FED ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
			if ( getExecType() == ExecType.FED ){
				sb.append( OPERAND_DELIMITOR );
				sb.append( _fedOutput.name() );
			}
		}
		
		return sb.toString();
	}
	
	private String getOpCode() {
		switch( _direction ) {
			case RowCol: return OPCODE_RC;
			case Col: return OPCODE_C;
			default: throw new RuntimeException("Unsupported aggregation direction: "+_direction);
		}
	}
}
