/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
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
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
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
			this.lps.setProperties(inputs, et, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob);
		}
		else {
			// when executing in CP, this lop takes an optional 3rd input (Weights)
			if ( input3 != null ) {
				this.addInput(input3);
				input3.addOutput(this);
			}
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties(inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob);
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

	@Override
	public String getInstructions(String input1, String input2, String input3, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "cm" );
		// value type for "order" is INT
		sb.append( OPERAND_DELIMITOR );
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input3 );
		sb.append( DATATYPE_PREFIX );
		sb.append( DataType.SCALAR );
		sb.append( VALUETYPE_PREFIX );
		sb.append( ValueType.INT );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(String input1, String input2, String output) {
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "cm" );
		// value type for "order" is INT
		sb.append( OPERAND_DELIMITOR ); 
		sb.append( input1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( DataType.SCALAR );
		sb.append( VALUETYPE_PREFIX );
		sb.append( ValueType.INT );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		
		return sb.toString();
	}
	
	@Override
	public String getInstructions(int input_index, int output_index) {
		
		// get label for scalar input -- the "order" for central moment.
		String order = this.getInputs().get(1).getOutputParameters().getLabel();
		/*
		 * if it is a literal, copy val, else surround with the label with
		 * ## symbols. these will be replaced at runtime.
		 */
		if(this.getInputs().get(1).getExecLocation() == ExecLocation.Data && 
				((Data)this.getInputs().get(1)).isLiteral())
			; // order = order;
		else
			order = "##" + order + "##";
		
		return getInstructions(input_index+"", order, output_index+"");
	}

}