/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class LeftIndex extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * Constructor to setup a LeftIndexing operation.
	 * Example: A[i:j, k:l] = B;
	 *      
	 * 
	 * @param input
	 * @param op
	 * @return 
	 * @throws LopsException
	 */
	
	private void init(Lop lhsMatrix, Lop rhsMatrix, Lop rowL, Lop rowU, Lop colL, Lop colU, ExecType et) throws LopsException {
		/*
		 * A[i:j, k:l] = B;
		 * B -> rhsMatrix
		 * A -> lhsMatrix
		 * i,j -> rowL, rowU
		 * k,l -> colL, colU
		 */
		this.addInput(lhsMatrix);
		this.addInput(rhsMatrix);
		this.addInput(rowL);
		this.addInput(rowU);
		this.addInput(colL);
		this.addInput(colU);
		
		lhsMatrix.addOutput(this);		
		rhsMatrix.addOutput(this);		
		rowL.addOutput(this);
		rowU.addOutput(this);
		colL.addOutput(this);
		colU.addOutput(this);

		boolean breaksAlignment = true;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			throw new LopsException(this.printErrorLocation() + "LeftIndexing lop is undefined for MR runtime");
		} 
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	public LeftIndex(
			Lop lhsInput, Lop rhsInput, Lop rowL, Lop rowU, Lop colL, Lop colU, DataType dt, ValueType vt, ExecType et)
			throws LopsException {
		super(Lop.Type.LeftIndex, dt, vt);
		init(lhsInput, rhsInput, rowL, rowU, colL, colU, et);
	}

	private String getOpcode() {
		return "leftIndex";
	}
	
	@Override
	public String getInstructions(String lhsInput, String rhsInput, String rowl, String rowu, String coll, String colu, String output) 
		throws LopsException 
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getOpcode() );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(lhsInput));
		sb.append( OPERAND_DELIMITOR );
		
		if ( getInputs().get(1).get_dataType() == DataType.SCALAR ) {
			sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()));
		}
		else {
			sb.append( getInputs().get(1).prepInputOperand(rhsInput));
		}
		sb.append( OPERAND_DELIMITOR );
		
		// rowl, rowu
		sb.append(getInputs().get(2).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(3).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		// rowl, rowu
		sb.append(getInputs().get(4).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		sb.append(getInputs().get(5).prepScalarInputOperand(getExecType()));
		sb.append( OPERAND_DELIMITOR );	
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

/*	@Override
	public String getInstructions(int input_index1, int input_index2, int input_index3, int input_index4, int input_index5, int output_index)
			throws LopsException {
		
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
*/
	@Override
	public String toString() {
		return "leftIndex";
	}

}
