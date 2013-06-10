package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Lop to perform data partitioning.
 */
public class DataPartition extends Lops 
{
	public static final String OPCODE = "partition"; 
	
	public DataPartition(Lops input, DataType dt, ValueType vt) throws LopsException 
	{
		super(Lops.Type.DataPartition, dt, vt);		
		this.addInput(input);
		input.addOutput(this);
		
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = true;
		
		// This lop executes only in DATA_PARTITION job
		lps.addCompatibility(JobType.DATA_PARTITION);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.MapAndReduce, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {
	
		return "DataPartition";
	}

	@Override
	public String getInstructions(int input_index, int output_index) throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lops.OPERAND_DELIMITOR );
		sb.append( OPCODE );
		sb.append( OPERAND_DELIMITOR );
		sb.append( input_index );
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
		
		return sb.toString();
	}
	
 
}