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

package org.apache.sysml.udf.lib;

import java.util.ArrayList;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysml.parser.Expression.DataType;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * This function is for testing purposes only. 
 */
public class EvalFunction extends PackageFunction 
{
	private static final long serialVersionUID = 1L;
	
	private Matrix _ret; 
	
	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		switch(pos) {
			case 0: return _ret;
			default:
				throw new RuntimeException("Invalid function output being requested");
		}
	}

	@Override
	public void execute(ExecutionContext ec) {
		String fname = ((Scalar)getFunctionInput(0)).getValue();
		MatrixObject in = ((Matrix) getFunctionInput(1)).getMatrixObject();
		ArrayList<String> inputs = new ArrayList<>(); inputs.add("A");
		ArrayList<String> outputs = new ArrayList<>(); outputs.add("B");
		
		ExecutionContext ec2 = ExecutionContextFactory.createContext(ec.getProgram());
		CPOperand inName = new CPOperand("TMP", org.apache.sysml.parser.Expression.ValueType.DOUBLE, DataType.MATRIX);
		ec2.setVariable("TMP", in);
		
		FunctionCallCPInstruction fcpi = new FunctionCallCPInstruction(
			null, fname, new CPOperand[]{inName}, inputs, outputs, "eval func");
		fcpi.processInstruction(ec2);
		
		MatrixObject out = (MatrixObject)ec2.getVariable("B");
		_ret = new Matrix(out, ValueType.Double);
	}

	@Override
	public void execute() {
		throw new NotImplementedException();
	}
}
