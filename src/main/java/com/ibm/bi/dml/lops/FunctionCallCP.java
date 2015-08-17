/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.lops;


import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecLocation;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


/**
 *
 */
public class FunctionCallCP extends Lop  
{
	
	private String _fnamespace;
	private String _fname;
	private String[] _outputs;
	private ArrayList<Lop> _outputLops = null;

	public FunctionCallCP(ArrayList<Lop> inputs, String fnamespace, String fname, String[] outputs, ArrayList<Hop> outputHops) throws HopsException, LopsException {
		this(inputs, fnamespace, fname, outputs);
		if(outputHops != null) {
			_outputLops = new ArrayList<Lop>();
			for(Hop h : outputHops) {
				_outputLops.add( h.constructLops() );
			}
		}
	}
	
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

	public ArrayList<Lop> getFunctionOutputs() {
		return _outputLops;
	}
	
	@Override
	public String toString() {
		return "function call: " + DMLProgram.constructFunctionKey(_fnamespace, _fname);
	}

	private String getInstructionsMultipleReturnBuiltins(String[] inputs, String[] outputs) {
		StringBuilder sb = new StringBuilder();
		sb.append("CP");
		sb.append(Lop.OPERAND_DELIMITOR); 
		sb.append(_fname.toLowerCase());
		sb.append(Lop.OPERAND_DELIMITOR); 
		for(int i=0; i< inputs.length; i++) {
			sb.append( getInputs().get(0).prepInputOperand(inputs[i]) );
			if ( i != inputs.length-1 )
				sb.append(Lop.OPERAND_DELIMITOR);
		}
		sb.append(Lop.OPERAND_DELIMITOR); 
		for(int i=0; i< _outputs.length; i++) {
			sb.append(_outputs[i]);
			if ( i != _outputs.length-1 )
				sb.append(Lop.OPERAND_DELIMITOR);
		}
		return sb.toString();
	}
	
	/**
	 * Method to generate instructions for external functions as well as builtin functions with multiple returns.
	 * Builtin functions have their namespace set to DMLProgram.INTERNAL_NAMESPACE ("_internal").
	 */
	@Override
	public String getInstructions(String[] inputs, String[] outputs) throws LopsException
	{		
		// Handle internal builtin functions
		if (_fnamespace.equalsIgnoreCase(DMLProgram.INTERNAL_NAMESPACE) ) {
			return getInstructionsMultipleReturnBuiltins(inputs, outputs);
		}
		
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



