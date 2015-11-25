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

package com.ibm.bi.dml.udf.lib;

//import com.ibm.bi.dml.packagesupport.Scalar;
//import com.ibm.bi.dml.packagesupport.FIO;

import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * Wrapper class for time invocation
 * 
 * time = externalFunction(Integer i) return (Double B) implemented in
 * (classname="com.ibm.bi.dml.udf.lib.TimeWrapper",exectype="mem");
 * 
 * t = time (1);
 * 
 * print( "Time in millsecs= " + t);
 * 
 */
public class TimeWrapper extends PackageFunction 
{
	
	private static final long serialVersionUID = 1L;

	private Scalar _ret;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if (pos == 0)
			return _ret;

		throw new PackageRuntimeException(
				"Invalid function output being requested");
	}

	@Override
	public void execute() {
		try {
			// int past = Integer.parseInt(((Scalar) getFunctionInput(0))
			// .getValue());
			long present = System.currentTimeMillis();
			_ret = new Scalar(ScalarValueType.Double, String.valueOf(present));
		} catch (Exception e) {
			throw new PackageRuntimeException(
					"Error executing external time function", e);
		}
	}

}
