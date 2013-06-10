package com.ibm.bi.dml.lops;

import java.util.HashSet;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.*;
import com.ibm.bi.dml.utils.LopsException;


/**
 * Lop to represent an combine operation -- used ONLY in the context of sort.
 */

public class CombineUnary extends Lops
{
	

	/**
	 * @param input - input lop
	 * @param op - operation type
	 */
	
	public CombineUnary(Lops input1, DataType dt, ValueType vt) 
	{
		super(Lops.Type.CombineUnary, dt, vt);	
		this.addInput(input1);
		input1.addOutput(this);
				
		/*
		 *  This lop can ONLY be executed as a SORT_KEYS job
		 *  CombineUnary instruction gets piggybacked into SORT_KEYS job
		 */
		boolean breaksAlignment = false;
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.SORT);
		this.lps.setProperties( inputs, ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}
	
	/**
	 * for debugging purposes. 
	 */
	
	public String toString()
	{
		return "combineunary";		
	}

	@Override
	public String getInstructions(int input_index1, int output_index) 
		throws LopsException
	{
		StringBuilder sb = new StringBuilder();
		sb.append( getExecType() );
		sb.append( Lops.OPERAND_DELIMITOR );
		sb.append( "combineunary" );
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
		
		return sb.toString();
	}

	public static CombineUnary constructCombineLop(Lops input1, 
			DataType dt, ValueType vt) {
		
		HashSet<Lops> set1 = new HashSet<Lops>();
		set1.addAll(input1.getOutputs());
			
		for (Lops lop  : set1) {
			if ( lop.type == Lops.Type.CombineUnary ) {
				return (CombineUnary)lop;
			}
		}
		
		CombineUnary comn = new CombineUnary(input1, dt, vt);
		comn.setAllPositions(input1.getBeginLine(), input1.getBeginColumn(), input1.getEndLine(), input1.getEndLine());
		return comn;
	}
	
 
}
