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

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * 
 *
 */
public class DynamicReadMatrixCP extends PackageFunction 
{	
	
	private static final long serialVersionUID = 1L;
	private Matrix _ret; 
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		return _ret;
	}

	@Override
	public void execute() 
	{
		try
		{
			String fname = ((Scalar) this.getFunctionInput(0)).getValue();
			Integer m = Integer.parseInt(((Scalar) this.getFunctionInput(1)).getValue());
			Integer n = Integer.parseInt(((Scalar) this.getFunctionInput(2)).getValue());
			String format = ((Scalar) this.getFunctionInput(3)).getValue();
			
			InputInfo ii = InputInfo.stringToInputInfo(format);
			OutputInfo oi = OutputInfo.BinaryBlockOutputInfo;
			
			MatrixBlock mbTmp = DataConverter.readMatrixFromHDFS(fname, ii, m, n, ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());			
			String fnameTmp = createOutputFilePathAndName("TMP");
			
			_ret = new Matrix(fnameTmp, m, n, ValueType.Double);
			_ret.setMatrixDoubleArray(mbTmp, oi, ii);
			
			//NOTE: The packagesupport wrapper creates a new MatrixObjectNew with the given 
			// matrix block. This leads to a dirty state of the new object. Hence, the resulting 
			// intermediate plan variable will be exported in front of MR jobs and during this export 
			// the format will be changed to binary block (the contract of external functions), 
			// no matter in which format the original matrix was.
		}
		catch(Exception e)
		{
			throw new RuntimeException("Error executing dynamic read of matrix",e);
		}
	}
}
