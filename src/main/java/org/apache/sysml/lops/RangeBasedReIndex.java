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

package org.apache.sysml.lops;

import org.apache.sysml.hops.AggBinaryOp.SparkAggType;
import org.apache.sysml.lops.LopProperties.ExecLocation;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;


public class RangeBasedReIndex extends Lop 
{
	
	/**
	 * Constructor to setup a RangeBasedReIndex operation.
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */
	
	private boolean forLeftIndexing = false;

	//optional attribute for spark exec type
	private SparkAggType _aggtype = SparkAggType.MULTI_BLOCK;

	public RangeBasedReIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU, Lop rowDim, Lop colDim, 
			DataType dt, ValueType vt, ExecType et, boolean forleft)
		throws LopsException 
	{
		super(Lop.Type.RangeReIndex, dt, vt);
		init(input, rowL, rowU, colL, colU, rowDim, colDim, dt, vt, et, forleft);
	}

	public RangeBasedReIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU, Lop rowDim, Lop colDim, 
			DataType dt, ValueType vt, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.RangeReIndex, dt, vt);
		init(input, rowL, rowU, colL, colU, rowDim, colDim, dt, vt, et, false);
	}

	public RangeBasedReIndex(Lop input, Lop rowL, Lop rowU, Lop colL, Lop colU, Lop rowDim, Lop colDim, 
			DataType dt, ValueType vt, SparkAggType aggtype, ExecType et)
		throws LopsException 
	{
		super(Lop.Type.RangeReIndex, dt, vt);
		_aggtype = aggtype;
		init(input, rowL, rowU, colL, colU, rowDim, colDim, dt, vt, et, false);
	}

	private void init(Lop inputMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU, Lop leftMatrixRowDim, 
			Lop leftMatrixColDim, DataType dt, ValueType vt, ExecType et, boolean forleft) 
	{	
		addInput(inputMatrix);
		addInput(rowL);
		addInput(rowU);
		addInput(colL);
		addInput(colU);
		addInput(leftMatrixRowDim);
		addInput(leftMatrixColDim);
		
		inputMatrix.addOutput(this);		
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);
		leftMatrixRowDim.addOutput(this);
		leftMatrixColDim.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			
			lps.addCompatibility(JobType.GMR);
			lps.addCompatibility(JobType.DATAGEN);
			lps.addCompatibility(JobType.MMCJ);
			lps.addCompatibility(JobType.MMRJ);
			lps.setProperties(inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob);
		} 
		else {
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
		
		forLeftIndexing=forleft;
	}
	
	private String getOpcode() {
		if(forLeftIndexing)
			return "rangeReIndexForLeft";
		else
			return "rangeReIndex";
	}
	
	@Override
	public String getInstructions(String input, String rowl, String rowu, String coll, String colu, String leftRowDim, String leftColDim, String output) 
	throws LopsException {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input));
		sb.append( OPERAND_DELIMITOR );
		
		// rowl, rowu
		sb.append( getInputs().get(1).prepScalarInputOperand(rowl));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(2).prepScalarInputOperand(rowu));
		sb.append( OPERAND_DELIMITOR );
		
		// coll, colu
		sb.append( getInputs().get(3).prepScalarInputOperand(coll));
		sb.append( OPERAND_DELIMITOR );
		sb.append( getInputs().get(4).prepScalarInputOperand(colu));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		
		if(getExecType() == ExecType.MR) {
			// following fields are added only when this lop is executed in MR (both for left & right indexing) 
			sb.append( OPERAND_DELIMITOR );
			
			sb.append( getInputs().get(5).prepScalarInputOperand(leftRowDim));
			sb.append( OPERAND_DELIMITOR );
			sb.append( getInputs().get(6).prepScalarInputOperand(leftColDim));
		}
		
		//in case of spark, we also compile the optional aggregate flag into the instruction.
		if( getExecType() == ExecType.SPARK ) {
			sb.append( OPERAND_DELIMITOR );
			sb.append( _aggtype );	
		}
		
		return sb.toString();
	}

	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int input_index5, int input_index6, int input_index7, int output_index)
			throws LopsException {
		/*
		 * Example: B = A[row_l:row_u, col_l:col_u]
		 * A - input matrix (input_index1)
		 * row_l - lower bound in row dimension
		 * row_u - upper bound in row dimension
		 * col_l - lower bound in column dimension
		 * col_u - upper bound in column dimension
		 * 
		 * Since row_l,row_u,col_l,col_u are scalars, values for input_index(2,3,4,5,6,7) 
		 * will be equal to -1. They should be ignored and the scalar value labels must
		 * be derived from input lops.
		 */
		String rowl = getInputs().get(1).prepScalarLabel();
		String rowu = getInputs().get(2).prepScalarLabel();
		String coll = getInputs().get(3).prepScalarLabel();
		String colu = getInputs().get(4).prepScalarLabel();

		String left_nrow = getInputs().get(5).prepScalarLabel();
		String left_ncol = getInputs().get(6).prepScalarLabel();
		
		return getInstructions(Integer.toString(input_index1), rowl, rowu, coll, colu, left_nrow, left_ncol, Integer.toString(output_index));
	}

	@Override
	public String toString() {
		if(forLeftIndexing)
			return "rangeReIndexForLeft";
		else
			return "rangeReIndex";
	}

}
