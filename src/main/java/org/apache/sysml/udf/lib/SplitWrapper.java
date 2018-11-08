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

import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.instructions.cp.StringObject;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.List;

/**
 * Wrapper class for split invocation
 * 
 * split = externalFunction(String s, String regex) return (list[String] out) implemented in
 * (classname="org.apache.sysml.udf.lib.SplitWrapper",exectype="mem");
 * 
 * out = split ("foo_goo_boo", "_"); 
 * for ( i in 1:3) { print(as.scalar(out[i])); }
 * 
 */
public class SplitWrapper extends PackageFunction {
	private static final long serialVersionUID = 1L;

	private List outputList;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if (pos == 0)
			return outputList;
		else
			throw new RuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() {
		String str = ((Scalar) getFunctionInput(0)).getValue();
		String regex = ((Scalar) getFunctionInput(1)).getValue();

		java.util.List<Data> outputData = new ArrayList<>();
		for(String part : str.split(regex)) {
			outputData.add(new StringObject(part));
		}
		outputList = new List(new ListObject(outputData));
	}

}
