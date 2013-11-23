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

/**
 * Lop to compute covariance between two 1D matrices
 * 
 */
public class CoVariance extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private void init(Lop input1, Lop input2, Lop input3, ExecType et) throws LopsException {
		/*
		 * When et = MR: covariance lop will have a single input lop, which
		 * denote the combined input data -- output of combinebinary, if unweighed;
		 * and output combineteriaty (if weighted).
		 * 
		 * When et = CP: covariance lop must have at least two input lops, which
		 * denote the two input columns on which covariance is computed. It also
		 * takes an optional third arguments, when weighted covariance is computed.
		 */
		this.addInput(input1);
		input1.addOutput(this);

		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		if ( et == ExecType.MR ) {
			lps.addCompatibility(JobType.CM_COV);
			this.lps.setProperties(inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			definesMRJob = false;
			if ( input2 == null ) {
				throw new LopsException(this.printErrorLocation() + "Invalid inputs to covariance lop.");
			}
			this.addInput(input2);
			input2.addOutput(this);
			
			if ( input3 != null ) {
				this.addInput(input3);
				input3.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	/**
	 * Constructor to perform covariance.
	 * input1 <- data 
	 * (prior to this lop, input vectors need to attached together using CombineBinary or CombineTertiary) 
	 * @throws LopsException 
	 */

	public CoVariance(Lop input1, DataType dt, ValueType vt) throws LopsException {
		this(input1, dt, vt, ExecType.MR);
	}

	public CoVariance(Lop input1, DataType dt, ValueType vt, ExecType et) throws LopsException {
		super(Lop.Type.CoVariance, dt, vt);
		init(input1, null, null, et);
	}
	
	public CoVariance(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) throws LopsException {
		this(input1, input2, null, dt, vt, et);
	}
	
	public CoVariance(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) throws LopsException {
		super(Lop.Type.CoVariance, dt, vt);
		init(input1, input2, input3, et);
	}

	@Override
	public String toString() {

		return "Operation = coVariance";
	}
	
	/**
	 * Function two generate CP instruction to compute unweighted covariance.
	 * input1 -> input column 1
	 * input2 -> input column 2
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "cov" );
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );

		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	/**
	 * Function two generate CP instruction to compute weighted covariance.
	 * input1 -> input column 1
	 * input2 -> input column 2
	 * input3 -> weights
	 */
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "cov" );
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(1).prepInputOperand(input2));
		sb.append( OPERAND_DELIMITOR );

		sb.append( getInputs().get(2).prepInputOperand(input3));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}

	/**
	 * Function to generate MR version of covariance instruction.
	 * input_index -> denote the "combined" input columns and weights, 
	 * when applicable.
	 */
	@Override
	public String getInstructions(int input_index, int output_index) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "cov" );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( getInputs().get(0).prepInputOperand(input_index));
		sb.append( OPERAND_DELIMITOR );
		
		sb.append ( this.prepInputOperand(output_index));
		
		return sb.toString();
	}

}