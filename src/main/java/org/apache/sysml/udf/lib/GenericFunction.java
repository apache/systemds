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

import java.io.IOException;

import org.apache.commons.lang.StringUtils;
import org.apache.sysml.api.ExternalUDFRegistration;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;

import scala.Function0;

public class GenericFunction extends PackageFunction {
	private static final long serialVersionUID = -195996547505886575L;
	String [] fnSignature;
	FunctionParameter [] returnVals;
	Function0<FunctionParameter []> scalaUDF;
	public String _functionName;
	public String _namespace;
	
	public void initialize() {
		if(_namespace != null && !_namespace.equals(".defaultNS")) {
			throw new RuntimeException("Expected the function in default namespace");
		}
		if(_functionName == null) {
			throw new RuntimeException("Expected the function name to be set");
		}
		if(fnSignature == null) {
			fnSignature = ExternalUDFRegistration.fnSignatureMapping().get(_functionName);
			scalaUDF = ExternalUDFRegistration.fnMapping().get(_functionName);
			ExternalUDFRegistration.udfMapping().put(_functionName, this);
		}
	}
	
	@Override
	public int getNumFunctionOutputs() {
		initialize();
		String retSignature = fnSignature[fnSignature.length -1];
		if(!retSignature.startsWith("("))
			return 1;
		else {
			return StringUtils.countMatches(retSignature, ",") + 1;
		}
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		initialize();
		if(returnVals == null || returnVals.length <= pos)
			throw new RuntimeException("Incorrect number of outputs or function not executed");
		return returnVals[pos];
	}

	@Override
	public void execute() {
		initialize();
		returnVals = scalaUDF.apply();
	}
	
	public Object getInput(String type, int pos) throws DMLRuntimeException, IOException {
		if(type.equals("Int") || type.equals("java.lang.Integer")) {
			return Integer.parseInt(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("Double") || type.equals("java.lang.Double")) {
			return Double.parseDouble(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("java.lang.String")) {
			return ((Scalar)getFunctionInput(pos)).getValue();
		}
		else if(type.equals("boolean") || type.equals("java.lang.Boolean")) {
			return Boolean.parseBoolean(((Scalar)getFunctionInput(pos)).getValue());
		}
		else if(type.equals("scala.Array[scala.Array[Double]]")) {
			return ((Matrix) getFunctionInput(pos)).getMatrixAsDoubleArray();
		}
		
		throw new RuntimeException("Unsupported type: " + type);
	}

}
