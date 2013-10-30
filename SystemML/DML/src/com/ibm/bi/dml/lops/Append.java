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


public class Append extends Lop
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public Append(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) 
	{
		super(Lop.Type.Append, dt, vt);
		init(input1, input2, input3, dt, vt, et);
	}
	
	public Append(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt) 
	{
		super(Lop.Type.Append, dt, vt);		
		init(input1, input2, input3, dt, vt, ExecType.MR);
	}
	
	public void init(Lop input1, Lop input2, Lop input3, DataType dt, ValueType vt, ExecType et) 
	{
		this.addInput(input1);
		input1.addOutput(this);

		this.addInput(input2);
		input2.addOutput(this);
		
		this.addInput(input3);
		input3.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		
		if ( et == ExecType.MR ) {
			//confirm this
			lps.addCompatibility(JobType.GMR);
			
			//TODO MB @Shirish: please review; disabled compatibility to reblock because even with 'setupDistributedCache' 
			//        (and partitioning information) in ReblockMR, this fails due to matrix block vs matrix cell
			//lps.addCompatibility(JobType.REBLOCK);
			this.lps.setProperties( inputs, et, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
		}
		else {
			lps.addCompatibility(JobType.INVALID);
			this.lps.setProperties( inputs, et, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
		}
	}
	
	@Override
	public String toString() {

		return " Append: ";
	}

	//called when append executes in MR
	public String getInstructions(int input_index1, int input_index2, int input_index3, int output_index) throws LopsException
	{
		return getInstructions(input_index1+"",
							   input_index2+"",
							   input_index3+"",
							   output_index+"");	
	}
	
	//....in CP
	public String getInstructions(String input_index1, String input_index2, String input_index3, String output_index) throws LopsException
	{
		String offsetString;
		String offsetLabel = this.getInputs().get(2).getOutputParameters().getLabel();
	
		if (getExecType() == ExecType.MR ) {
			if (this.getInputs().get(2).getExecLocation() == ExecLocation.Data
				&& ((Data) this.getInputs().get(2)).isLiteral())
				offsetString = "" + offsetLabel;
			else
				offsetString = "##" + offsetLabel + "##";
		}
		else {
			offsetString = "" + offsetLabel;
		}
	
		StringBuilder sb = new StringBuilder();
		sb.append( this.lps.execType );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "append" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() ); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index2 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(1).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(1).get_valueType() ); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() ); 
		sb.append( OPERAND_DELIMITOR );
		sb.append( offsetString );
		
		return sb.toString();
	}
}
