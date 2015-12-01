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

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.PackageRuntimeException;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * External function (type CP) used within the PerfTestTool in order to
 * measure the general behavior of package support.
 *
 */
public class PerfTestExtFunctCP extends PackageFunction 
{	
	
	private static final long   serialVersionUID = 1L;
	private static final String OUTPUT_FILE      = "PerfTestExtFunctOutput";
	
	private static IDSequence   _idSeq   = new IDSequence(); 
	private Matrix              _ret     = null; 
	private String              _baseDir = null;
	
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
	
	public void setBaseDir(String dir)
	{
		_baseDir = dir;
	}

	@Override
	public void execute() 
	{
		try
		{
			long id = _idSeq.getNextID();
			
			Matrix in = (Matrix) this.getFunctionInput(0);
			double [][] aIn = in.getMatrixAsDoubleArray();
			
			int rows = aIn.length;
			int cols = aIn[0].length;
			
			String dir = _baseDir + "/" + OUTPUT_FILE+id;
			
			//copy and write output data 
			MatrixBlock mb = new MatrixBlock(rows,cols,false);
			for(int i=0; i < rows; i++)
				for(int j=0; j < cols; j++)
					mb.setValue(i, j, aIn[i][j]);

			_ret = new Matrix(dir, rows, cols, ValueType.Double);
			_ret.setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);	
		}
		catch(Exception e)
		{
			throw new PackageRuntimeException("Error executing generic test extfunct.", e);
		}
	}

}
