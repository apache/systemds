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

package org.apache.sysds.runtime.instructions.fed;

import java.util.HashMap;
import java.util.LinkedHashMap;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.SimpleOperator;

public class ParameterizedBuiltinFEDInstruction extends ComputationFEDInstruction {

	protected final LinkedHashMap<String, String> params;
	
	protected ParameterizedBuiltinFEDInstruction(Operator op,
		LinkedHashMap<String, String> paramsMap, CPOperand out, String opcode, String istr)
	{
		super(FEDType.ParameterizedBuiltin, op, null, null, out, opcode, istr);
		params = paramsMap;
	}
	
	public HashMap<String,String> getParameterMap() { 
		return params; 
	}
	
	public String getParam(String key) {
		return getParameterMap().get(key);
	}
	
	public static LinkedHashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		LinkedHashMap<String,String> paramMap = new LinkedHashMap<>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static ParameterizedBuiltinFEDInstruction parseInstruction ( String str ) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 
	
		// process remaining parts and build a hash map
		LinkedHashMap<String,String> paramsMap = constructParameterMap(parts);
	
		// determine the appropriate value function
		ValueFunction func = null;
		if( opcode.equalsIgnoreCase("replace") ) {
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinFEDInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unsupported opcode (" + opcode + ") for ParameterizedBuiltinFEDInstruction.");
		}
	}
	
	@Override 
	public void processInstruction(ExecutionContext ec) {
		String opcode = getOpcode();
		if ( opcode.equalsIgnoreCase("replace") ) {
			//similar to unary federated instructions, get federated input
			//execute instruction, and derive federated output matrix
			MatrixObject mo = getTarget(ec);
			FederatedRequest fr1 = FederationUtils.callInstruction(instString, output,
				new CPOperand[]{getTargetOperand()}, new long[]{mo.getFedMapping().getID()});
			mo.getFedMapping().execute(fr1);
			
			//derive new fed mapping for output
			MatrixObject out = ec.getMatrixObject(output);
			out.getDataCharacteristics().set(mo.getDataCharacteristics());
			out.setFedMapping(mo.getFedMapping().copyWithNewID(fr1.getID()));
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}
	}
	
	public MatrixObject getTarget(ExecutionContext ec) {
		return ec.getMatrixObject(params.get("target"));
	}
	
	private CPOperand getTargetOperand() {
		return new CPOperand(params.get("target"), ValueType.FP64, DataType.MATRIX);
	}
}
