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

package org.apache.sysml.udf.lib;

import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.PackageRuntimeException;
import org.apache.sysml.udf.Matrix.ValueType;

public class DynamicProjectMatrixCP extends PackageFunction
{
	
	private static final long serialVersionUID = 1L;	
	private static final String OUTPUT_FILE = "DynProjectMatrixWrapperOutput2D";
	
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
			Matrix mD = (Matrix) this.getFunctionInput(0);
			Matrix mC = (Matrix) this.getFunctionInput(1);
			MatrixBlock mbD = mD.getMatrixObject().acquireRead();
			MatrixBlock mbC = mC.getMatrixObject().acquireRead();
			int rows = mbC.getNumColumns();
			int cols = mbC.getNumColumns();
			
			String dir = createOutputFilePathAndName( OUTPUT_FILE );
			
			MatrixBlock mb = null;
			
			if( mbD.getNumColumns()==1 ) //VECTOR
			{
				cols=1;
				mb = new MatrixBlock(rows,cols,false);
				
				for(int i=0; i < rows; i++)
				{
					int ix1 = (int)mbC.quickGetValue(0, i)-1;
					double val = mbD.quickGetValue(ix1, 0);
					mb.quickSetValue(i, 0, val);	
				}	
			}
			else //MATRIX
			{
				mb = new MatrixBlock(rows,cols,false);
				
				for(int i=0; i < rows; i++)
				{
					int ix1 = (int)mbC.quickGetValue(0, i)-1;
					for(int j=0; j < cols; j++)
					{
						int ix2 = (int)mbC.quickGetValue(0, j)-1;
						double val = mbD.quickGetValue(ix1, ix2);
						mb.quickSetValue(i, j, val);	
					}			
				}
			}
			_ret = new Matrix(dir, rows, cols, ValueType.Double);			
			_ret.setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
	
			mD.getMatrixObject().release();
			mC.getMatrixObject().release();
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing dynamic project of matrix", e);
		}
	}	
}
