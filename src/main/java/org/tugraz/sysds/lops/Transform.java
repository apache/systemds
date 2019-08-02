/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.lops;

 
import org.tugraz.sysds.lops.LopProperties.ExecType;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;


/*
 * Lop to perform transpose/vector to diag operations
 * This lop can change the keys and hence break alignment.
 */
public class Transform extends Lop
{
	public enum OperationTypes {
		Transpose,
		Diag,
		Reshape,
		Sort,
		Rev
	}
	
	private OperationTypes operation = null;
	private boolean _bSortIndInMem = false;
	private boolean _outputEmptyBlock = true;
	private int _numThreads = 1;
	
	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) {
		this(input, op, dt, vt, et, 1);
	}
	
	public Transform(Lop[] inputs, Transform.OperationTypes op, DataType dt, ValueType vt, boolean outputEmptyBlock, ExecType et) {
		this(inputs, op, dt, vt, et, 1);
		_outputEmptyBlock = outputEmptyBlock;
	}
	
	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k)  {
		super(Lop.Type.Transform, dt, vt);
		init(new Lop[]{input}, op, dt, vt, et);
		_numThreads = k;
	}
	
	public Transform(Lop[] inputs, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et, int k)  {
		super(Lop.Type.Transform, dt, vt);
		init(inputs, op, dt, vt, et);
		_numThreads = k;
	}

	public Transform(Lop input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et, boolean bSortIndInMem) {
		super(Lop.Type.Transform, dt, vt);
		_bSortIndInMem = bSortIndInMem;
		init(new Lop[]{input}, op, dt, vt, et);
	}
	
	public Transform(Lop[] inputs, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et, boolean bSortIndInMem) {
		super(Lop.Type.Transform, dt, vt);
		_bSortIndInMem = bSortIndInMem;
		init(inputs, op, dt, vt, et);
	}
	
	private void init (Lop[] input, Transform.OperationTypes op, DataType dt, ValueType vt, ExecType et) 
	{
		operation = op;
		
		for(Lop in : input) {
			this.addInput(in);
			in.addOutput(this);
		}
		
		lps.setProperties( inputs, et);
	}

	@Override
	public String toString() {

		return " Operation: " + operation;
	}

	/**
	 * method to get operation type
	 * @return operaton type
	 */
	 
	public OperationTypes getOperationType()
	{
		return operation;
	}

	private String getOpcode() {
		switch(operation) {
		case Transpose:
			// Transpose a matrix
			return "r'";
		
		case Rev:
			// Transpose a matrix
			return "rev";
		
		case Diag:
			// Transform a vector into a diagonal matrix
			return "rdiag";
		
		case Reshape:
			// Transform a vector into a diagonal matrix
			return "rshape";
		
		case Sort:
			// Transform a matrix into a sorted matrix 
			return "rsort";
		
		default:
			throw new UnsupportedOperationException(this.printErrorLocation() + "Instruction is not defined for Transform operation " + operation);
		}
	}
	
	//CP instructions
	
	@Override
	public String getInstructions(String input1, String output) {
		//opcodes: r', rev, rdiag
		return getInstructions(input1, 1, output);
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
		StringBuilder sb = new StringBuilder();
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
		
		if( getExecType()==ExecType.CP && operation == OperationTypes.Transpose ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _numThreads );
		}
		if( getExecType()==ExecType.SPARK && operation == OperationTypes.Reshape ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _outputEmptyBlock );
		}
		if( getExecType()==ExecType.SPARK && operation == OperationTypes.Sort ){
			sb.append( OPERAND_DELIMITOR );
			sb.append( _bSortIndInMem );
		}
		
		return sb.toString();
	}
}
