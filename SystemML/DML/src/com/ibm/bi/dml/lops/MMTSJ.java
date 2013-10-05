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
 * Lop to perform transpose-identity operation (t(X)%*%X or X%*%t(X)),
 * used to represent CP and MR instruction but in case of MR there is
 * an additional Aggregate at the reducers.
 */
public class MMTSJ extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum MMTSJType {
		NONE,
		LEFT,
		RIGHT
	}
	
	private MMTSJType _type = null;
	
	public MMTSJ(Lop input1, DataType dt, ValueType vt, ExecType et, MMTSJType type) 
	{
		super(Lop.Type.MMTSJ, dt, vt);		
		addInput(input1);
		input1.addOutput(this);
		_type = type;
		 
		boolean breaksAlignment = true; //if result keys (matrix indexes) different 
		boolean aligner = false; //if groups multiple inputs by key (e.g., group)
		boolean definesMRJob = (et == ExecType.MR); //if requires its own MR job 
		ExecLocation el = (et == ExecType.MR) ? ExecLocation.Map : ExecLocation.ControlProgram;
		
		lps.addCompatibility(JobType.GMR);
		//lps.addCompatibility(JobType.MMTSJ);
		lps.setProperties( inputs, et, el, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "Operation = MMTSJ";
	}

	//TODO CP inst
	
	@Override
	public String getInstructions(int input_index1, int output_index)
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( "tsmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _type );
		
		return sb.toString();
	}

	public String getInstructions(String input_index1, String output_index) throws LopsException
	{	
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( "tsmm" );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index1 );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( get_dataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( get_valueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _type );
		
		return sb.toString();
	}
 
 
}