/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;


import java.util.ArrayList;

import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.Program;


/**
 *
 */
public class FunctionCallCP extends Lop  
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _fnamespace;
	private String _fname;
	private String[] _outputs;

	public FunctionCallCP(ArrayList<Lop> inputs, String fnamespace, String fname, String[] outputs) 
	{
		super(Lop.Type.FunctionCallCP, DataType.UNKNOWN, ValueType.UNKNOWN);	
		//note: data scalar in order to prevent generation of redundant createvar, rmvar
		
		_fnamespace = fnamespace;
		_fname = fname;
		_outputs = outputs;
		
		//wire inputs
		for( Lop in : inputs )
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
		inst.append(Lop.OPERAND_DELIMITOR); 
		inst.append("extfunct");
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fnamespace);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fname);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(inputs.length);
		inst.append(Lop.OPERAND_DELIMITOR);
		//inst.append(outputs.length);  TODO function output dataops (phase 3)
		inst.append(_outputs.length);
		for( String in : inputs )
		{
			inst.append(Lop.OPERAND_DELIMITOR);
			inst.append(in);
		}
		
		for( String out : _outputs )
		{
			inst.append(Lop.OPERAND_DELIMITOR);
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



