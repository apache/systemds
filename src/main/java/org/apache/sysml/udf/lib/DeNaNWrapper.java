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


import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.PackageRuntimeException;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * Wrapper class to deNaN matrices by replacing all NaNs with zeros,
 * made by modifying <code>OrderWrapper.java</code>
 */
public class DeNaNWrapper extends PackageFunction 
{
	
	private static final long serialVersionUID = 1L;
	private static  final String OUTPUT_FILE = "TMP";

	//return matrix
	private Matrix ret;

	@Override
	public int getNumFunctionOutputs() 
	{
		return 1;	
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) 
	{	
		if(pos == 0)
			return ret;
		
		throw new PackageRuntimeException("Invalid function output being requested");
	}

	@Override
	public void execute() 
	{ 
		try 
		{
			Matrix inM = (Matrix) getFunctionInput(0);
			double [][] inData = inM.getMatrixAsDoubleArray();
			for (int i = 0; i < inData.length; i++) {
				for (int j = 0; j < inData[i].length; j++) {
					if (Double.isNaN (inData [i][j])) {
						inData [i][j] = 0.0;
			}   }   }
			//create and copy output matrix		
			String dir = createOutputFilePathAndName( OUTPUT_FILE );	
			ret = new Matrix( dir, inM.getNumRows(), inM.getNumCols(), ValueType.Double );
			ret.setMatrixDoubleArray(inData);
		} 
		catch (Exception e) 
		{
			throw new PackageRuntimeException("Error executing external removeNaN function", e);
		}
	}
}
