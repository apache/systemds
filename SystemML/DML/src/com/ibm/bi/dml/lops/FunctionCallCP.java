package com.ibm.bi.dml.lops;


import java.util.ArrayList;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.utils.LopsException;


/**
 *
 */
public class FunctionCallCP extends Lops  
{
	private String _fnamespace;
	private String _fname;
	private String[] _outputs;

	public FunctionCallCP(ArrayList<Lops> inputs, String fnamespace, String fname, String[] outputs) 
	{
		super(Lops.Type.FunctionCallCP, DataType.UNKNOWN, ValueType.UNKNOWN);	
		//note: data scalar in order to prevent generation of redundant createvar, rmvar
		
		_fnamespace = fnamespace;
		_fname = fname;
		_outputs = outputs;
		
		//wire inputs
		for( Lops in : inputs )
		{
			addInput( in );
			in.addOutput( this );
		}
			
		//lop properties: always in CP
		boolean breaksAlignment = false; 
		boolean aligner = false;
		boolean definesMRJob = false;
		lps.addCompatibility(JobType.INVALID);
		this.lps.setProperties(inputs, ExecType.CP, ExecLocation.ControlProgram, breaksAlignment, aligner, definesMRJob );
	}

	@Override
	public String toString() {

		return "function call: " + _fname+Program.KEY_DELIM+_fnamespace;

	}

	@Override
	public String getInstructions(String[] inputs, String[] outputs) throws LopsException
	{		
		/**
		 * Instruction format extFunct:::[FUNCTION NAMESPACE]:::[FUNCTION NAME]:::[num input params]:::[num output params]:::[list of delimited input params ]:::[list of delimited ouput params]
		 * These are the "bound names" for the inputs / outputs.  For example, out1 = ns::foo(in1, in2) yields
		 * extFunct:::ns:::foo:::2:::1:::in1:::in2:::out1
		 * 
		 */

		StringBuilder inst = new StringBuilder();
		
		inst.append("CP");
		inst.append(Lops.OPERAND_DELIMITOR); 
		inst.append("extfunct");
		inst.append(Lops.OPERAND_DELIMITOR);
		inst.append(_fnamespace);
		inst.append(Lops.OPERAND_DELIMITOR);
		inst.append(_fname);
		inst.append(Lops.OPERAND_DELIMITOR);
		inst.append(inputs.length);
		inst.append(Lops.OPERAND_DELIMITOR);
		//inst.append(outputs.length);  TODO function output dataops (phase 3)
		inst.append(_outputs.length);
		for( String in : inputs )
		{
			inst.append(Lops.OPERAND_DELIMITOR);
			inst.append(in);
		}
		
		for( String out : _outputs )
		{
			inst.append(Lops.OPERAND_DELIMITOR);
			inst.append(out);
		}		
		/* TODO function output dataops (phase 3)
		for( String out : outputs )
		{
			inst.append(Lops.OPERAND_DELIMITOR);
			inst.append(out);
		}
		*/
		
		return inst.toString();				
	}
}



