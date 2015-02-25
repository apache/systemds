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
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;


/**
 * Lop to perform data partitioning.
 */
public class DataPartition extends Lop 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final String OPCODE = "partition"; 
	
	private PDataPartitionFormat _pformat = null;
	
	public DataPartition(Lop input, DataType dt, ValueType vt, ExecType et, PDataPartitionFormat pformat) 
		throws LopsException 
	{
		super(Lop.Type.DataPartition, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		_pformat = pformat;
		
		//setup lop properties
		ExecLocation eloc = (et==ExecType.MR)? ExecLocation.MapAndReduce : ExecLocation.ControlProgram;
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = (et==ExecType.MR);
		lps.addCompatibility(JobType.DATA_PARTITION);
		lps.setProperties( inputs, et, eloc, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "DataPartition";
	}

	//CP instruction generation
	@Override
	public String getInstructions(String input_index, String output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _pformat.toString() );
		
		return sb.toString();
	}
	
	//MR instruction generation
	@Override
	public String getInstructions(int input_index, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lop.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getInputs().get(0).getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getInputs().get(0).getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( output_index );
		sb.append( DATATYPE_PREFIX );
		sb.append( getDataType() );
		sb.append( VALUETYPE_PREFIX );
		sb.append( getValueType() );
		sb.append( OPERAND_DELIMITOR );
		sb.append( _pformat.toString() );
		
		return sb.toString();
	}
}