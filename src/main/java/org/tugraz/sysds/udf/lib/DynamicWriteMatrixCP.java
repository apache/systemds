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

package org.tugraz.sysds.udf.lib;

import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.matrix.MatrixCharacteristics;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.udf.FunctionParameter;
import org.tugraz.sysds.udf.Matrix;
import org.tugraz.sysds.udf.PackageFunction;
import org.tugraz.sysds.udf.Scalar;
import org.tugraz.sysds.udf.Scalar.ScalarValueType;

@Deprecated
public class DynamicWriteMatrixCP extends PackageFunction 
{	
	
	private static final long serialVersionUID = 1L;
	private Scalar _success;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		return _success;
	}

	@Override
	public void execute() 
	{
		boolean success = false;
		
		try
		{
			Matrix mat = (Matrix) this.getFunctionInput(0);
			String fname = ((Scalar) this.getFunctionInput(1)).getValue();
			String format = ((Scalar) this.getFunctionInput(2)).getValue();
			
			MatrixObject mo = mat.getMatrixObject();
			MatrixCharacteristics mc = mo.getMatrixCharacteristics();
			OutputInfo oi = OutputInfo.stringToOutputInfo(format);
			
			MatrixBlock mb = mo.acquireRead();
			DataConverter.writeMatrixToHDFS(mb, fname, oi, mc);			
			mo.release();
			
			success = true;
		}
		catch(Exception e)
		{
			throw new RuntimeException("Error executing dynamic write of matrix",e);
		}
		
		_success = new Scalar(ScalarValueType.Boolean,String.valueOf(success));
	}
}
