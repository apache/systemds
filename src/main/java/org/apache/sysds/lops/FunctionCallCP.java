/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.lops;


import java.util.ArrayList;

import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;

public class FunctionCallCP extends Lop
{	
	private String _fnamespace;
	private String _fname;
	private String[] _inputNames;
	private String[] _outputNames;
	private ArrayList<Lop> _outputLops = null;
	private boolean _opt;

	public FunctionCallCP(ArrayList<Lop> inputs, String fnamespace, String fname, String[] inputNames,
		String[] outputNames, ArrayList<Hop> outputHops, boolean opt, ExecType et) {
		this(inputs, fnamespace, fname, inputNames, outputNames, et);
		if(outputHops != null) {
			_outputLops = new ArrayList<>();
			setLevel();
			for(Hop h : outputHops) {
				Lop outputLop = h.constructLops();
				_outputLops.add( outputLop );
				addOutput(outputLop);
				// Update the output level if necessary for correct instruction ordering
				if(outputLop.getLevel() <= getLevel()) {
					outputLop.updateLevel(getLevel()+1);
				}
			}
		}
		_opt = opt;
	}
	
	public FunctionCallCP(ArrayList<Lop> inputs, String fnamespace, String fname, String[] inputNames, String[] outputNames, ExecType et) 
	{
		super(Lop.Type.FunctionCallCP, DataType.UNKNOWN, ValueType.UNKNOWN);
		//note: data scalar in order to prevent generation of redundant createvar, rmvar
		
		_fnamespace = fnamespace;
		_fname = fname;
		_inputNames = inputNames;
		_outputNames = outputNames;
		
		//wire inputs
		for( Lop in : inputs ) {
			addInput( in );
			in.addOutput( this );
		}
		
		//lop properties: always in CP
		lps.setProperties(inputs, et);
	}

	public ArrayList<Lop> getFunctionOutputs() {
		return _outputLops;
	}
	
	public boolean requiresOutputCreateVar() {
		return !_fname.equalsIgnoreCase(Builtins.REMOVE.getName());
	}
	
	@Override
	public String toString() {
		return "function call: " + DMLProgram.constructFunctionKey(_fnamespace, _fname);
	}

	private String getInstructionsMultipleReturnBuiltins(String[] inputs, String[] outputs) {
		StringBuilder sb = new StringBuilder();
		sb.append(getExecType());
		
		sb.append(Lop.OPERAND_DELIMITOR); 
		sb.append(_fname.toLowerCase());
		
		for(int i=0; i< inputs.length; i++) {
			sb.append(Lop.OPERAND_DELIMITOR);
			sb.append( getInputs().get(i).prepInputOperand(inputs[i]) );
		}
		
		for(int i=0; i< _outputNames.length; i++) {
			sb.append(Lop.OPERAND_DELIMITOR);
			sb.append(_outputNames[i]);
		}
		
		return sb.toString();
	}
	
	/**
	 * Method to generate instructions for external functions as well as builtin functions with multiple returns.
	 * Builtin functions have their namespace set to DMLProgram.INTERNAL_NAMESPACE ("_internal").
	 */
	@Override
	public String getInstructions(String[] inputs, String[] outputs)
	{		
		// Handle internal builtin functions
		if (_fnamespace.equalsIgnoreCase(DMLProgram.INTERNAL_NAMESPACE) ) {
			return getInstructionsMultipleReturnBuiltins(inputs, outputs);
		}

		StringBuilder inst = new StringBuilder();
		inst.append(getExecType());
		
		inst.append(Lop.OPERAND_DELIMITOR); 
		inst.append(FunctionOp.OPCODE);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fnamespace);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_fname);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_opt);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(inputs.length);
		inst.append(Lop.OPERAND_DELIMITOR);
		inst.append(_outputNames.length);
		
		for(int i=0; i<inputs.length; i++) {
			inst.append(Lop.OPERAND_DELIMITOR);
			inst.append(_inputNames[i]);
			inst.append("=");
			inst.append( getInputs().get(i).prepInputOperand(inputs[i]) );
		}

		// TODO function output dataops (phase 3) - take 'outputs' into account
		for( String out : _outputNames ) {
			inst.append(Lop.OPERAND_DELIMITOR);
			inst.append(out);
		}

		return inst.toString();
	}
}
