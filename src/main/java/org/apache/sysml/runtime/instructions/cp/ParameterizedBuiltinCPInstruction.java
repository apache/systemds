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

package org.apache.sysml.runtime.instructions.cp;

import java.io.IOException;
import java.util.HashMap;

import org.apache.wink.json4j.JSONException;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.functionobjects.ParameterizedBuiltin;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.instructions.mr.GroupedAggregateInstruction;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.Operator;
import org.apache.sysml.runtime.matrix.operators.SimpleOperator;
import org.apache.sysml.runtime.transform.DataTransform;


public class ParameterizedBuiltinCPInstruction extends ComputationCPInstruction 
{
	private int arity;
	protected HashMap<String,String> params;
	
	public ParameterizedBuiltinCPInstruction(Operator op, HashMap<String,String> paramsMap, CPOperand out, String opcode, String istr )
	{
		super(op, null, null, out, opcode, istr);
		_cptype = CPINSTRUCTION_TYPE.ParameterizedBuiltin;
		params = paramsMap;
	}

	public int getArity() {
		return arity;
	}
	
	public HashMap<String,String> getParameterMap() { return params; }
	
	public static HashMap<String, String> constructParameterMap(String[] params) {
		// process all elements in "params" except first(opcode) and last(output)
		HashMap<String,String> paramMap = new HashMap<String,String>();
		
		// all parameters are of form <name=value>
		String[] parts;
		for ( int i=1; i <= params.length-2; i++ ) {
			parts = params[i].split(Lop.NAME_VALUE_SEPARATOR);
			paramMap.put(parts[0], parts[1]);
		}
		
		return paramMap;
	}
	
	public static ParameterizedBuiltinCPInstruction parseInstruction ( String str ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// first part is always the opcode
		String opcode = parts[0];
		// last part is always the output
		CPOperand out = new CPOperand( parts[parts.length-1] ); 

		// process remaining parts and build a hash map
		HashMap<String,String> paramsMap = constructParameterMap(parts);

		// determine the appropriate value function
		ValueFunction func = null;
		if ( opcode.equalsIgnoreCase("cdf") ) {
			if ( paramsMap.get("dist") == null ) 
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("invcdf") ) {
			if ( paramsMap.get("dist") == null ) 
				throw new DMLRuntimeException("Invalid distribution: " + str);
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode, paramsMap.get("dist"));
			// Determine appropriate Function Object based on opcode
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equalsIgnoreCase("groupedagg")) {
			// check for mandatory arguments
			String fnStr = paramsMap.get("fn");
			if ( fnStr == null ) 
				throw new DMLRuntimeException("Function parameter is missing in groupedAggregate.");
			if ( fnStr.equalsIgnoreCase("centralmoment") ) {
				if ( paramsMap.get("order") == null )
					throw new DMLRuntimeException("Mandatory \"order\" must be specified when fn=\"centralmoment\" in groupedAggregate.");
			}
			
			Operator op = GroupedAggregateInstruction.parseGroupedAggOperator(fnStr, paramsMap.get("order"));
			return new ParameterizedBuiltinCPInstruction(op, paramsMap, out, opcode, str);
		}
		else if(   opcode.equalsIgnoreCase("rmempty") 
				|| opcode.equalsIgnoreCase("replace") 
				|| opcode.equalsIgnoreCase("rexpand") ) 
		{
			func = ParameterizedBuiltin.getParameterizedBuiltinFnObject(opcode);
			return new ParameterizedBuiltinCPInstruction(new SimpleOperator(func), paramsMap, out, opcode, str);
		}
		else if ( opcode.equals("transform")) {
			return new ParameterizedBuiltinCPInstruction(null, paramsMap, out, opcode, str);
		}
		else {
			throw new DMLRuntimeException("Unknown opcode (" + opcode + ") for ParameterizedBuiltin Instruction.");
		}

	}

	@Override 
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		
		String opcode = getOpcode();
		ScalarObject sores = null;
		
		if ( opcode.equalsIgnoreCase("cdf")) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("invcdf")) {
			SimpleOperator op = (SimpleOperator) _optr;
			double result =  op.fn.execute(params);
			sores = new DoubleObject(result);
			ec.setScalarOutput(output.getName(), sores);
		} 
		else if ( opcode.equalsIgnoreCase("groupedagg") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get(Statement.GAGG_TARGET));
			MatrixBlock groups = ec.getMatrixInput(params.get(Statement.GAGG_GROUPS));
			MatrixBlock weights= null;
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				weights = ec.getMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
			int ngroups = -1;
			if ( params.get(Statement.GAGG_NUM_GROUPS) != null) {
				ngroups = (int) Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS));
			}
			
			// compute the result
			int k = Integer.parseInt(params.get("k")); //num threads
			MatrixBlock soresBlock = groups.groupedAggOperations(target, weights, new MatrixBlock(), ngroups, _optr, k);
			
			ec.setMatrixOutput(output.getName(), soresBlock);
			// release locks
			target = groups = weights = null;
			ec.releaseMatrixInput(params.get(Statement.GAGG_TARGET));
			ec.releaseMatrixInput(params.get(Statement.GAGG_GROUPS));
			if ( params.get(Statement.GAGG_WEIGHTS) != null )
				ec.releaseMatrixInput(params.get(Statement.GAGG_WEIGHTS));
			
		}
		else if ( opcode.equalsIgnoreCase("rmempty") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			MatrixBlock select = params.containsKey("select")? ec.getMatrixInput(params.get("select")):null;
			
			// compute the result
			String margin = params.get("margin");
			MatrixBlock soresBlock = null;
			if( margin.equals("rows") )
				soresBlock = target.removeEmptyOperations(new MatrixBlock(), true, select);
			else if( margin.equals("cols") ) 
				soresBlock = target.removeEmptyOperations(new MatrixBlock(), false, select);
			else
				throw new DMLRuntimeException("Unspupported margin identifier '"+margin+"'.");
			
			//release locks
			ec.setMatrixOutput(output.getName(), soresBlock);
			ec.releaseMatrixInput(params.get("target"));
			if (params.containsKey("select"))
				ec.releaseMatrixInput(params.get("select"));
		}
		else if ( opcode.equalsIgnoreCase("replace") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			
			// compute the result
			double pattern = Double.parseDouble( params.get("pattern") );
			double replacement = Double.parseDouble( params.get("replacement") );
			MatrixBlock ret = (MatrixBlock) target.replaceOperations(new MatrixBlock(), pattern, replacement);
			
			//release locks
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("rexpand") ) {
			// acquire locks
			MatrixBlock target = ec.getMatrixInput(params.get("target"));
			
			// compute the result
			double maxVal = Double.parseDouble( params.get("max") );
			boolean dirVal = params.get("dir").equals("rows");
			boolean cast = Boolean.parseBoolean(params.get("cast"));
			boolean ignore = Boolean.parseBoolean(params.get("ignore"));
			MatrixBlock ret = (MatrixBlock) target.rexpandOperations(new MatrixBlock(), maxVal, dirVal, cast, ignore);
			
			//release locks
			ec.setMatrixOutput(output.getName(), ret);
			ec.releaseMatrixInput(params.get("target"));
		}
		else if ( opcode.equalsIgnoreCase("transform")) {
			MatrixObject mo = (MatrixObject) ec.getVariable(params.get("target"));
			MatrixObject out = (MatrixObject) ec.getVariable(output.getName());
			
			try {
				JobReturn jt = DataTransform.cpDataTransform(this, new MatrixObject[] { mo } , new MatrixObject[] {out} );
				out.updateMatrixCharacteristics(jt.getMatrixCharacteristics(0));
			} catch (IllegalArgumentException e) {
				throw new DMLRuntimeException(e);
			} catch (IOException e) {
				throw new DMLRuntimeException(e);
			} catch (JSONException e) {
				throw new DMLRuntimeException(e);
			}
		}
		else {
			throw new DMLRuntimeException("Unknown opcode : " + opcode);
		}		
	}
}
