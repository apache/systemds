package dml.lops;

import java.util.HashSet;

import dml.lops.LopProperties.ExecLocation;
import dml.lops.LopProperties.ExecType;
import dml.lops.compile.JobType;
import dml.parser.Expression.*;
import dml.utils.LopsException;

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
		this.lps.setProperties( ExecType.MR, ExecLocation.Map, breaksAlignment, aligner, definesMRJob );
	}
	
	/**
	 * for debugging purposes. 
	 */
	
	public String toString()
	{
		return "combineunary";		
	}

	@Override
	public String getInstructions(int input_index1, int output_index) throws LopsException
	{
		String inst = new String(getExecType() + Lops.OPERAND_DELIMITOR);
		inst += "combineunary" + OPERAND_DELIMITOR + 
		        input_index1 + VALUETYPE_PREFIX + this.getInputs().get(0).get_valueType() + OPERAND_DELIMITOR + 
		        output_index + VALUETYPE_PREFIX + this.get_valueType() ;
		
		return inst;
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
		return comn;
	}
	
 
}
