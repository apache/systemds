/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;

public class ZeroOut  extends Lop 
{

	
	private void init(Lop inputMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU, long rowDim, long colDim, DataType dt, ValueType vt, ExecType et) {
		this.addInput(inputMatrix);
		this.addInput(rowL);
		this.addInput(rowU);
		this.addInput(colL);
		this.addInput(colU);
		
		inputMatrix.addOutput(this);		
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			this.lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		} 
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	public ZeroOut(
			Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU, long rowDim, long colDim, DataType dt, ValueType vt)
			throws LopsException {
		super(Lop.Type.ZeroOut, dt, vt);
		init(input, rowL, rowU, colL, colU,  rowDim, colDim, dt, vt, ExecType.MR);
	}

	public ZeroOut(
			Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU, long rowDim, long colDim, DataType dt, ValueType vt, ExecType et)
			throws LopsException {
		super(Lop.Type.ZeroOut, dt, vt);
		init(input, rowL, rowU, colL, colU, rowDim, colDim, dt, vt, et);
	}
	
	private String getOpcode() {
		
			return "zeroOut";
	}
	
	@Override
	public String getInstructions(String input, String rowl, String rowu, String coll, String colu, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(0).prepInputOperand(input));
		sb.append( OPERAND_DELIMITOR ); 
		
		// rowl, rowu
		sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
				
		// coll, colu
		sb.append( getInputs().get(3).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(4).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int input_index5, int output_index)
			throws LopsException {
		/*
		 * Example: B = A[row_l:row_u, col_l:col_u]
		 * A - input matrix (input_index1)
		 * row_l - lower bound in row dimension
		 * row_u - upper bound in row dimension
		 * col_l - lower bound in column dimension
		 * col_u - upper bound in column dimension
		 * 
		 * Since row_l,row_u,col_l,col_u are scalars, values for input_index(2,3,4,5) 
		 * will be equal to -1. They should be ignored and the scalar value labels must
		 * be derived from input lops.
		 */
		
		return getInstructions(Integer.toString(input_index1), input_index2+"", input_index3+"", input_index4+"", input_index5+"", Integer.toString(output_index));
	}

	@Override
	public String toString() {
		return "ZeroOut";
	}
}
