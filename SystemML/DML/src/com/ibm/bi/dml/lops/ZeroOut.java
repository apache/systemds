package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;

public class ZeroOut  extends Lops {

	private void init(Lops inputMatrix, Lops rowL, Lops rowU, Lops colL, Lops colU, long rowDim, long colDim, DataType dt, ValueType vt, ExecType et) {
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
			lps.addCompatibility(JobType.RAND);
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
			Lops input, Lops rowL, Lops rowU, Lops colL, Lops colU, long rowDim, long colDim, DataType dt, ValueType vt)
			throws LopsException {
		super(Lops.Type.ZeroOut, dt, vt);
		init(input, rowL, rowU, colL, colU,  rowDim, colDim, dt, vt, ExecType.MR);
	}

	public ZeroOut(
			Lops input, Lops rowL, Lops rowU, Lops colL, Lops colU, long rowDim, long colDim, DataType dt, ValueType vt, ExecType et)
			throws LopsException {
		super(Lops.Type.ZeroOut, dt, vt);
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
		sb.append( input );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR ); 
		sb.append( rowl );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( rowu );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(2).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(2).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( coll );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(3).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(3).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( colu );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(4).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(4).get_valueType() );
		sb.append( OPERAND_DELIMITOR ); 
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() ); 
		
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
		String rowl = this.getInputs().get(1).getOutputParameters().getLabel();
		if (this.getInputs().get(1).getExecLocation() != ExecLocation.Data
				|| !((Data) this.getInputs().get(1)).isLiteral())
			rowl = "##" + rowl + "##";
		String rowu = this.getInputs().get(2).getOutputParameters().getLabel();
		if (this.getInputs().get(2).getExecLocation() != ExecLocation.Data
				|| !((Data) this.getInputs().get(2)).isLiteral())
			rowu = "##" + rowu + "##";
		String coll = this.getInputs().get(3).getOutputParameters().getLabel();
		if (this.getInputs().get(3).getExecLocation() != ExecLocation.Data
				|| !((Data) this.getInputs().get(3)).isLiteral())
			coll = "##" + coll + "##";
		String colu = this.getInputs().get(4).getOutputParameters().getLabel();
		if (this.getInputs().get(4).getExecLocation() != ExecLocation.Data
				|| !((Data) this.getInputs().get(4)).isLiteral())
			colu = "##" + colu + "##";
		
		return getInstructions(Integer.toString(input_index1), rowl, rowu, coll, colu, Integer.toString(output_index));
	}

	@Override
	public String toString() {
		return "ZeroOut";
	}
}
