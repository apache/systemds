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
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.instructions.InstructionUtils;


/*
 * Lop to perform transpose/vector to diag operations
 * This lop can change the keys and hence break alignment.
 */
public class Transform extends Lop
{
	private ReOrgOp _operation = null;
	private boolean _bSortIndInMem = false;
	private boolean _outputEmptyBlock = true;
	private int _numThreads = 1;
	
	public Transform(Lop input, ReOrgOp op, DataType dt, ValueType vt, ExecType et) {
		this(input, op, dt, vt, et, 1);
	}
	
	public Transform(Lop[] inputs, ReOrgOp op, DataType dt, ValueType vt, boolean outputEmptyBlock, ExecType et) {
		this(inputs, op, dt, vt, et, 1);
		_outputEmptyBlock = outputEmptyBlock;
	}
	
	public Transform(Lop input, ReOrgOp op, DataType dt, ValueType vt, ExecType et, int k)  {
		super(Lop.Type.Transform, dt, vt);
		init(new Lop[]{input}, op, dt, vt, et);
		_numThreads = k;
	}
	
	public Transform(Lop[] inputs, ReOrgOp op, DataType dt, ValueType vt, ExecType et, int k)  {
		super(Lop.Type.Transform, dt, vt);
		init(inputs, op, dt, vt, et);
		_numThreads = k;
	}

	public Transform(Lop input, ReOrgOp op, DataType dt, ValueType vt, ExecType et, boolean bSortIndInMem) {
		super(Lop.Type.Transform, dt, vt);
		_bSortIndInMem = bSortIndInMem;
		init(new Lop[]{input}, op, dt, vt, et);
	}
	
	public Transform(Lop[] inputs, ReOrgOp op, DataType dt, ValueType vt, ExecType et, boolean bSortIndInMem) {
		super(Lop.Type.Transform, dt, vt);
		_bSortIndInMem = bSortIndInMem;
		init(inputs, op, dt, vt, et);
	}

	public Transform(Lop[] inputs, ReOrgOp op, DataType dt, ValueType vt, ExecType et, boolean bSortIndInMem, int k) {
		super(Lop.Type.Transform, dt, vt);
		_bSortIndInMem = bSortIndInMem;
		_numThreads = k;
		init(inputs, op, dt, vt, et);
	}
	
	private void init (Lop[] input, ReOrgOp op, DataType dt, ValueType vt, ExecType et) {
		_operation = op;
		for(Lop in : input) {
			addInput(in);
			in.addOutput(this);
		}
		lps.setProperties(inputs, et);
	}

	@Override
	public String toString() {
		return " Operation: " + _operation;
	}

	/**
	 * method to get operation type
	 * @return operaton type
	 */
	 
	public ReOrgOp getOp() {
		return _operation;
	}

	private String getOpcode() {
		switch(_operation) {
			case TRANS:
				// Transpose a matrix
				return Opcodes.TRANSPOSE.toString();
			
			case REV:
				// Transpose a matrix
				return Opcodes.REV.toString();

			case ROLL:
				return Opcodes.ROLL.toString();

			case DIAG:
				// Transform a vector into a diagonal matrix
				return Opcodes.DIAG.toString();
			
			case RESHAPE:
				// Transform a vector into a diagonal matrix
				return Opcodes.RESHAPE.toString();
			
			case SORT:
				// Transform a matrix into a sorted matrix 
				return Opcodes.SORT.toString();
			
			default:
				throw new UnsupportedOperationException(printErrorLocation() 
					+ "Instruction is not defined for Transform operation " + _operation);
		}
	}
	
	//CP instructions
	
	@Override
	public String getInstructions(String input1, String output) {
		//opcodes: r', rev, rdiag
		return getInstructions(input1, 1, output);
	}

	@Override
	public String getInstructions(String input1, String input2, String output) {
		//opcodes: roll
		return getInstructions(input1, 2, output);
	}

	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String output) {
		//opcodes: rsort
		return getInstructions(input1, 4, output);
	}
	
	@Override
	public String getInstructions(String input1, String input2, String input3, String input4, String input5, String output) {
		//opcodes: rshape
		return getInstructions(input1, 5, output);
	}
	
	private String getInstructions(String input1, int numInputs, String output) {
		StringBuilder sb = InstructionUtils.getStringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input1));
		
		//rows, cols, byrow
		for( int i = 1; i < numInputs; i++ ) {
			Lop ltmp = getInputs().get(i);
			sb.append( OPERAND_DELIMITOR );
			sb.append( ltmp.prepScalarInputOperand(getExecType()));
		}
		
		//output
		sb.append( OPERAND_DELIMITOR );
		sb.append( this.prepOutputOperand(output));
		
		if( (getExecType()==ExecType.CP || getExecType()==ExecType.FED)
			&& (_operation == ReOrgOp.TRANS || _operation == ReOrgOp.REV || _operation == ReOrgOp.SORT) ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
			if ( getExecType()==ExecType.FED ) {
				sb.append( OPERAND_DELIMITOR );
				sb.append( _fedOutput.name() );
			}
		}
		else if( getExecType()==ExecType.SPARK && _operation == ReOrgOp.RESHAPE ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _outputEmptyBlock );
		}
		else if( getExecType()==ExecType.SPARK && _operation == ReOrgOp.SORT ){
			sb.append( OPERAND_DELIMITOR );
			sb.append( _bSortIndInMem );
		}
		
		return sb.toString();
	}
}
