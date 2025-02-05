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
 * Lop to perform ternary operation. All inputs must be matrices or vectors. 
 * For example, this lop is used in evaluating A = ctable(B,C,W)
 * 
 * Currently, this lop is used only in case of CTABLE functionality.
 */

public class Ctable extends Lop 
{
	private final boolean _ignoreZeros;
	private final boolean _outputEmptyBlocks;
	private final int _numThreads;
	
	public enum OperationTypes {
		CTABLE_TRANSFORM,
		CTABLE_TRANSFORM_SCALAR_WEIGHT,
		CTABLE_TRANSFORM_HISTOGRAM, 
		CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM,
		CTABLE_EXPAND_SCALAR_WEIGHT,
		INVALID;
		public boolean hasSecondInput() {
			return this == CTABLE_TRANSFORM
				|| this == CTABLE_EXPAND_SCALAR_WEIGHT
				|| this == CTABLE_TRANSFORM_SCALAR_WEIGHT;
		}
		public boolean hasThirdInput() {
			return this == CTABLE_TRANSFORM
				|| this == CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM;
		}
	}
	
	OperationTypes operation;
	

	public Ctable(Lop[] inputLops, OperationTypes op, DataType dt, ValueType vt, ExecType et, int k) {
		this(inputLops, op, dt, vt, false, true, et, k);
	}
	
	public Ctable(Lop[] inputLops, OperationTypes op, DataType dt, ValueType vt, boolean ignoreZeros, boolean outputEmptyBlocks, ExecType et, int k) {
		super(Lop.Type.Ctable, dt, vt);
		init(inputLops, op, et);
		_ignoreZeros = ignoreZeros;
		_outputEmptyBlocks = outputEmptyBlocks;
		_numThreads = k;
	}
	
	private void init(Lop[] inputLops, OperationTypes op, ExecType et) {
		operation = op;
		
		for(int i=0; i < inputLops.length; i++) {
			addInput(inputLops[i]);
			inputLops[i].addOutput(this);
		}
		
		lps.setProperties(inputs, et);
	}
	
	@Override
	public String toString() {
	
		return " Operation: " + operation;

	}

	public static OperationTypes findCtableOperationByInputDataTypes(DataType dt1, DataType dt2, DataType dt3) 
	{
		if ( dt1 == DataType.MATRIX ) {
			if (dt2 == DataType.MATRIX && dt3 == DataType.SCALAR) {
				// F = ctable(A,B) or F = ctable(A,B,1)
				return OperationTypes.CTABLE_TRANSFORM_SCALAR_WEIGHT;
			} else if (dt2 == DataType.SCALAR && dt3 == DataType.SCALAR) {
				// F=ctable(A,1) or F = ctable(A,1,1)
				return OperationTypes.CTABLE_TRANSFORM_HISTOGRAM;
			} else if (dt2 == DataType.SCALAR && dt3 == DataType.MATRIX) {
				// F=ctable(A,1,W)
				return OperationTypes.CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM;
			} else {
				// F=ctable(A,B,W)
				return OperationTypes.CTABLE_TRANSFORM;
			}
		}
		else {
			return OperationTypes.INVALID;
		}
	}

	/**
	 * method to get operation type
	 * @return operation type
	 */
	 
	public OperationTypes getOperationType() {
		return operation;
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String output)
	{
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		if( operation != Ctable.OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT )
			sb.append( Opcodes.CTABLE.toString() );
		else
			sb.append( Opcodes.CTABLEEXPAND.toString() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).getDataType() == DataType.SCALAR ?
			getInputs().get(0).prepScalarInputOperand(getExecType()) :
			getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(1).getDataType() == DataType.SCALAR ?
			getInputs().get(1).prepScalarInputOperand(getExecType()) :
			getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(2).getDataType() == DataType.SCALAR ?
			getInputs().get(2).prepScalarInputOperand(getExecType()) :
			getInputs().get(2).prepInputOperand(input3));
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().size() > 3 ) {
			sb.append(getInputs().get(3).getOutputParameters().getLabel());
			sb.append(LITERAL_PREFIX);
			sb.append((getInputs().get(3).getType() == Type.Data && ((Data)getInputs().get(3)).isLiteral()) );
			sb.append( OPERAND_DELIMITOR );

			sb.append(getInputs().get(4).getOutputParameters().getLabel());
			sb.append(LITERAL_PREFIX);
			sb.append((getInputs().get(4).getType() == Type.Data && ((Data)getInputs().get(4)).isLiteral()) );
			sb.append( OPERAND_DELIMITOR );
		}
		else {
			sb.append(-1);
			sb.append(LITERAL_PREFIX);
			sb.append(true);
			sb.append( OPERAND_DELIMITOR );
			
			sb.append(-1);
			sb.append(LITERAL_PREFIX);
			sb.append(true);
			sb.append( OPERAND_DELIMITOR ); 
		}
		sb.append( prepOutputOperand(output));
		
		sb.append( OPERAND_DELIMITOR );
		sb.append( _ignoreZeros );
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _outputEmptyBlocks );
		}
		else {
			sb.append( OPERAND_DELIMITOR );
			sb.append(_numThreads);
		}
		
		return sb.toString();
	}

	public static OperationTypes getOperationType(String opcode)
	{
		OperationTypes op = null;
		
		if( opcode.equals("ctabletransform") )
			op = OperationTypes.CTABLE_TRANSFORM;
		else if( opcode.equals("ctabletransformscalarweight") )
			op = OperationTypes.CTABLE_TRANSFORM_SCALAR_WEIGHT;
		else if( opcode.equals("ctableexpandscalarweight") )
			op = OperationTypes.CTABLE_EXPAND_SCALAR_WEIGHT;
		else if( opcode.equals("ctabletransformhistogram") )
			op = OperationTypes.CTABLE_TRANSFORM_HISTOGRAM;
		else if( opcode.equals("ctabletransformweightedhistogram") )
			op = OperationTypes.CTABLE_TRANSFORM_WEIGHTED_HISTOGRAM;
		else
			throw new UnsupportedOperationException("Tertiary operation code is not defined: " + opcode);
		
		return op;
	}
}