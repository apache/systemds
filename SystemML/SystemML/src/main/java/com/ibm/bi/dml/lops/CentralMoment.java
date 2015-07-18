/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;


/**
 * Lop to perform cross product operation
 */
public class CentralMoment extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	/**
	 * Constructor to perform central moment.
	 * input1 <- data (weighted or unweighted)
	 * input2 <- order (integer: 0, 2, 3, or 4)
	 */

	private void init(Lop input1, Lop input2, Lop input3, ExecType et) {
		this.addInput(input1);
		this.addInput(input2);
		input1.addOutput(this);
		input2.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		if ( et == ExecType.MR ) {
			definesMRJob = true;
			lps.addCompatibility(JobType.CM_COV);
			lps.setProperties(inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else { //CP/SPARK
			// when executing in CP, this lop takes an optional 3rd input (Weights)
			if ( input3 != null ) {
				this.addInput(input3);
				input3.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
		}
	}
	
	public CentralMoment(Lop input1, Lop input2, DataType dt, ValueType vt) {
		this(input1, input2, null, dt, vt, ExecType.MR);
	}

	public CentralMoment(Lop input1, Lop input2, DataType dt, ValueType vt, ExecType et) {
		this(input1, input2, null, dt, vt, et);
	}

	public CentralMoment(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) {
		this(input1, input2, input3, dt, vt, ExecType.MR);
	}

	public CentralMoment(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) {
		super(Lop.Type.CentralMoment, dt, vt);
		init(input1, input2, input3, et);
	}

	@Override
	public String toString() {

		return "Operation = CentralMoment";
	}

	/**
	 * Function to generate CP centralMoment instruction for weighted operation.
	 * 
	 * input1: data
	 * input2: weights
	 * input3: order
	 */
	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( "cm" );
		sb.append( OPERAND_DELIMITOR );
		
		// Input data
		sb.append( getInputs().get(0).prepInputOperand(input1) );
		sb.append( OPERAND_DELIMITOR );
		
		// Weights
		sb.append( getInputs().get(1).prepInputOperand(input2) );
		sb.append( OPERAND_DELIMITOR );
		
		// Order
		sb.append( getInputs().get(2).prepScalarInputOperand(getExecType()) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( prepOutputOperand(output));
		
		return sb.toString();
	}
	
	/**
	 * Function to generate CP centralMoment instruction for unweighted operation.
	 * 
	 * input1: data
	 * input2: order (not used, and order is derived internally!)
	 */
	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		
		sb.append( "cm" );
		sb.append( OPERAND_DELIMITOR ); 
		
		// Input data (can be weighted or unweighted)
		sb.append( getInputs().get(0).prepInputOperand(input1));
		sb.append( OPERAND_DELIMITOR );
		
		// Order
		sb.append( getInputs().get(1).prepScalarInputOperand(getExecType()) );
		sb.append( OPERAND_DELIMITOR );
		
		sb.append( this.prepOutputOperand(output));
		
		return sb.toString();
	}
	
	/**
	 * Function to generate MR central moment instruction, invoked from
	 * <code>Dag.java:getAggAndOtherInstructions()</code>. This function
	 * is used for both weighted and unweighted operations.
	 * 
	 * input_index: data (in case of weighted: data combined via combinebinary)
	 * output_index: produced output
	 * 
	 * The order for central moment is derived internally in the function.
	 */
	@Override
	public String getInstructions(int input_index, int output_index) {
		return getInstructions(input_index+"", "", output_index+"");
	}

}