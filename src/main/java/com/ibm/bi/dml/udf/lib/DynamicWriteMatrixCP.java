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

import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * 
 *
 */
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
			throw new PackageRuntimeException("Error executing dynamic write of matrix",e);
		}
		
		_success = new Scalar(ScalarValueType.Boolean,String.valueOf(success));
	}
}
