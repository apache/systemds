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

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.Matrix.ValueType;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

/**
 * 
 *
 */
public class DynamicReadMatrixRcCP extends PackageFunction 
{	
	
	private static final long serialVersionUID = 1L;
	private Matrix _ret;
	private Scalar _rc;
	
	@Override
	public int getNumFunctionOutputs() 
	{
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{
		if(pos == 0)
			return _ret;
		
		if(pos == 1)
			return _rc;
		
		throw new PackageRuntimeException("Invalid function output being requested");
		
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
			
			String fnameTmp = createOutputFilePathAndName("TMP");
			_ret = new Matrix(fnameTmp, m, n, ValueType.Double);

			MatrixBlock mbTmp = DataConverter.readMatrixFromHDFS(fname, ii, m, n, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);			
			_ret.setMatrixDoubleArray(mbTmp, oi, ii);
			_rc = new Scalar(ScalarValueType.Integer, "0");
			
			//NOTE: The packagesupport wrapper creates a new MatrixObjectNew with the given 
			// matrix block. This leads to a dirty state of the new object. Hence, the resulting 
			// intermediate plan variable will be exported in front of MR jobs and during this export 
			// the format will be changed to binary block (the contract of external functions), 
			// no matter in which format the original matrix was.
		}
		catch(Exception e)
		{
			_rc = new Scalar(ScalarValueType.Integer, "1");
//			throw new PackageRuntimeException("Error executing dynamic read of matrix",e);
		}
	}
}
